#include "colmap/sfm/global_mapper.h"

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/estimators/triangulation.h"
#include "colmap/math/union_find.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>

#include <boost/property_tree/json_parser.hpp>

namespace colmap {
namespace {

void LoadPosePriors(const GlobalMapperOptions& options,
                    Reconstruction& reconstruction) {
  boost::property_tree::ptree root;
  boost::property_tree::read_json(options.pose_prior_path.string(), root);
  const auto prior_dir =
      std::filesystem::absolute(options.pose_prior_path).parent_path();
  const auto image_dir = options.image_path.empty()
                             ? prior_dir
                             : std::filesystem::absolute(options.image_path);
  FlatHashMap<std::string, Rigid3d> cams_from_world;
  for (const auto& [_, entry] : root.get_child("frames")) {
    const auto path =
        (prior_dir / entry.get<std::string>("file_path")).lexically_normal();
    const auto& rows = entry.get_child("transform_matrix");
    THROW_CHECK_EQ(rows.size(), 4) << "Invalid pose matrix for " << path;
    Eigen::Matrix4d matrix;
    int row = 0;
    for (const auto& [_, values] : rows) {
      THROW_CHECK_EQ(values.size(), 4) << "Invalid pose matrix for " << path;
      int col = 0;
      for (const auto& [_, value] : values) {
        matrix(row, col++) = value.get_value<double>();
      }
      ++row;
    }
    THROW_CHECK(matrix.allFinite()) << "Non-finite pose for " << path;
    THROW_CHECK(matrix.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-6))
        << "Invalid homogeneous pose for " << path;
    Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
    THROW_CHECK((rotation.transpose() * rotation)
                    .isApprox(Eigen::Matrix3d::Identity(), 1e-4))
        << "Invalid rotation for " << path;
    THROW_CHECK_LT(std::abs(rotation.determinant() - 1), 1e-4)
        << "Invalid rotation determinant for " << path;
    // NeRF/OpenGL cameras look down -Z with +Y up; COLMAP uses +Z and -Y.
    rotation.col(1) *= -1;
    rotation.col(2) *= -1;
    const Rigid3d world_from_cam(Eigen::Quaterniond(rotation).normalized(),
                                 matrix.topRightCorner<3, 1>());
    THROW_CHECK(
        cams_from_world.emplace(path.generic_string(), Inverse(world_from_cam))
            .second)
        << "Duplicate pose prior for " << path;
  }

  FlatHashMap<frame_t, Rigid3d> rigs_from_world;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    const auto path = (image_dir / image.Name()).lexically_normal();
    const auto it = cams_from_world.find(path.generic_string());
    THROW_CHECK(it != cams_from_world.end())
        << "Missing pose prior for image " << image.Name();
    Frame frame = reconstruction.Frame(image.FrameId());
    frame.SetCamFromWorld(image.CameraId(), it->second);
    const auto [rig_it, inserted] =
        rigs_from_world.emplace(image.FrameId(), frame.RigFromWorld());
    THROW_CHECK(inserted || rig_it->second.ToMatrix().isApprox(
                                frame.RigFromWorld().ToMatrix(), 1e-4))
        << "Inconsistent pose priors for rig frame " << image.FrameId();
  }
  for (const auto& [frame_id, rig_from_world] : rigs_from_world) {
    reconstruction.Frame(frame_id).SetRigFromWorld(rig_from_world);
    reconstruction.RegisterFrame(frame_id);
  }
  LOG(INFO) << "Loaded pose priors for " << reconstruction.NumRegImages()
            << " images; skipping rotation averaging and global positioning";
}

void TriangulateTracks(const GlobalMapperOptions& options,
                       Reconstruction& reconstruction) {
  EstimateTriangulationOptions tri_options;
  tri_options.min_tri_angle = DegToRad(options.min_tri_angle_deg);
  tri_options.ransac_options.max_error =
      DegToRad(options.max_angular_reproj_error_deg);
  tri_options.ransac_options.random_seed = options.random_seed;
  for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
    const Track track = reconstruction.Point3D(point3D_id).track;
    std::vector<Eigen::Vector2d> points;
    std::vector<Rigid3d> cams_from_world;
    std::vector<const Camera*> cameras;
    for (const auto& el : track.Elements()) {
      const auto& image = reconstruction.Image(el.image_id);
      points.push_back(image.Point2D(el.point2D_idx).xy);
      cams_from_world.push_back(image.CamFromWorld());
      cameras.push_back(image.CameraPtr());
    }
    std::vector<char> inlier_mask;
    Eigen::Vector3d xyz;
    const bool success = EstimateTriangulation(
        tri_options, points, cams_from_world, cameras, &inlier_mask, &xyz);
    reconstruction.DeletePoint3D(point3D_id);
    if (!success) {
      continue;
    }
    Track inlier_track;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        inlier_track.AddElement(track.Element(i));
      }
    }
    if (inlier_track.Length() >=
        static_cast<size_t>(options.track_min_num_views_per_track)) {
      reconstruction.AddPoint3D(xyz, std::move(inlier_track));
    }
  }
  LOG(INFO) << "Triangulated " << reconstruction.NumPoints3D()
            << " points from prior poses";
}

bool RunBundleAdjustment(const BundleAdjustmentOptions& options,
                         Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no registered images";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no 3D points to optimize";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto ba = CreateDefaultBundleAdjuster(options, ba_config, reconstruction);

  return ba->Solve()->IsSolutionUsable();
}

}  // namespace

RotationEstimatorOptions GlobalMapperOptions::RotationAveraging() const {
  RotationEstimatorOptions opts = rotation_averaging;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
  }
  return opts;
}

GlobalPositionerOptions GlobalMapperOptions::GlobalPositioning() const {
  GlobalPositionerOptions opts = global_positioning;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  opts.solver_options.num_threads = num_threads;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
    opts.use_parameter_block_ordering = false;
  }
  return opts;
}

BundleAdjustmentOptions GlobalMapperOptions::BundleAdjustment() const {
  BundleAdjustmentOptions opts = bundle_adjustment;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  if (opts.ceres) {
    opts.ceres->solver_options.num_threads = num_threads;
    opts.ceres->gpu_index = ba_gpu_index;
  }
  if (opts.caspar) {
    opts.caspar->gpu_index = ba_gpu_index;
  }
  return opts;
}

IncrementalTriangulator::Options GlobalMapperOptions::Retriangulation() const {
  IncrementalTriangulator::Options opts = retriangulation;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
  }
  return opts;
}

GlobalMapper::GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache)
    : database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  pose_graph_ = std::make_shared<class PoseGraph>();
  pose_graph_->Load(*database_cache_->CorrespondenceGraph());
}

std::shared_ptr<Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Read pose priors from the database cache.
  const std::vector<PosePrior>& pose_priors = database_cache_->PosePriors();

  // First pass: solve rotation averaging on all frames, then filter outlier
  // pairs by rotation error and de-register frames outside the largest
  // connected component.
  RotationEstimatorOptions custom_options = options;
  custom_options.filter_unregistered = false;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  // Second pass: re-solve on registered frames only to refine rotations
  // after outlier removal.
  custom_options.filter_unregistered = true;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  VLOG(1) << reconstruction_->NumRegImages() << " / "
          << reconstruction_->NumImages()
          << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks(const GlobalMapperOptions& options) {
  using Observation = std::pair<image_t, point2D_t>;
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);

  // Build keypoints map from registered images.
  NodeHashMap<image_t, std::vector<Eigen::Vector2d>> image_id_to_keypoints;
  for (const auto image_id : reconstruction_->RegImageIds()) {
    const auto& image = reconstruction_->Image(image_id);
    std::vector<Eigen::Vector2d> points;
    points.reserve(image.NumPoints2D());
    for (const auto& point2D : image.Points2D()) {
      points.push_back(point2D.xy);
    }
    image_id_to_keypoints.emplace(image_id, std::move(points));
  }

  auto corr_graph = database_cache_->CorrespondenceGraph();

  // Union all matching observations.
  UnionFind<Observation, PairHash> uf;
  FeatureMatches matches;
  for (const auto& [pair_id, edge] : pose_graph_->ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    THROW_CHECK(image_id_to_keypoints.count(image_id1))
        << "Missing keypoints for image " << image_id1;
    THROW_CHECK(image_id_to_keypoints.count(image_id2))
        << "Missing keypoints for image " << image_id2;
    corr_graph->ExtractMatchesBetweenImages(image_id1, image_id2, matches);
    for (const auto& match : matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    }
  }

  // Group observations by their root.
  uf.Compress();
  NodeHashMap<Observation, std::vector<Observation>, PairHash> track_map;
  for (const auto& [obs, root] : uf.Parents()) {
    track_map[root].push_back(obs);
  }
  LOG(INFO) << "Established " << track_map.size() << " tracks from "
            << uf.Parents().size() << " observations";

  // Validate tracks, check consistency, and collect valid ones with lengths.
  NodeHashMap<point3D_t, Point3D> candidate_points3D;
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;

  for (const auto& [track_id, observations] : track_map) {
    NodeHashMap<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : observations) {
      const Eigen::Vector2d& xy =
          image_id_to_keypoints.at(image_id).at(feature_id);

      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          const double sq_threshold =
              options.track_intra_image_consistency_threshold *
              options.track_intra_image_consistency_threshold;
          if ((existing_xy - xy).squaredNorm() > sq_threshold) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          ++discarded_counter;
          break;
        }
        it->second.push_back(xy);
      } else {
        image_id_set[image_id].push_back(xy);
      }
      point3D.track.AddElement(image_id, feature_id);
    }

    if (!is_consistent) continue;

    const size_t num_images = image_id_set.size();
    if (num_images < static_cast<size_t>(options.track_min_num_views_per_track))
      continue;

    const point3D_t point3D_id = next_point3D_id++;
    track_lengths.emplace_back(point3D.track.Length(), point3D_id);
    candidate_points3D.emplace(point3D_id, std::move(point3D));
  }

  LOG(INFO) << "Kept " << candidate_points3D.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";

  // Sort tracks by length (descending) and select for problem.
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  NodeHashMap<image_t, size_t> tracks_per_image;
  size_t images_left = image_id_to_keypoints.size();
  const size_t max_num_tracks =
      static_cast<size_t>(options.keep_max_num_tracks);
  for (const auto& [track_length, point3D_id] : track_lengths) {
    // Stop once the global track budget is exhausted. As tracks are sorted by
    // decreasing length, this keeps the longest tracks and bounds memory usage.
    if (reconstruction_->NumPoints3D() >= max_num_tracks) break;

    auto& point3D = candidate_points3D.at(point3D_id);

    // Check if any image in this track still needs more observations.
    const bool should_add = std::any_of(
        point3D.track.Elements().begin(),
        point3D.track.Elements().end(),
        [&](const auto& obs) {
          return tracks_per_image[obs.image_id] <=
                 static_cast<size_t>(options.track_required_tracks_per_view);
        });
    if (!should_add) continue;

    // Update image counts.
    for (const auto& obs : point3D.track.Elements()) {
      auto& count = tracks_per_image[obs.image_id];
      if (count == static_cast<size_t>(options.track_required_tracks_per_view))
        --images_left;
      ++count;
    }

    // Add track after updating counts so we can move.
    reconstruction_->AddPoint3D(point3D_id, std::move(point3D));

    if (images_left == 0) break;
  }

  LOG(INFO) << "Before filtering: " << candidate_points3D.size()
            << ", after filtering: " << reconstruction_->NumPoints3D();
}

bool GlobalMapper::GlobalPositioning(const GlobalPositionerOptions& options,
                                     double max_angular_reproj_error_deg,
                                     double max_normalized_reproj_error,
                                     double min_tri_angle_deg) {
  if (!RunGlobalPositioning(options, *pose_graph_, *reconstruction_)) {
    return false;
  }

  // Filter tracks based on the estimation
  ObservationManager obs_manager(*reconstruction_);

  // First pass: use relaxed threshold (2x) for cameras without prior focal.
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      2.0 * max_angular_reproj_error_deg,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::ANGULAR);

  // Second pass: apply strict threshold for cameras with prior focal length.
  const double max_angular_error_rad = DegToRad(max_angular_reproj_error_deg);
  std::vector<std::pair<image_t, point2D_t>> obs_to_delete;
  for (const auto point3D_id : reconstruction_->Point3DIds()) {
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    const auto& point3D = reconstruction_->Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      const auto& image = reconstruction_->Image(track_el.image_id);
      const auto& camera = *image.CameraPtr();
      if (!camera.has_prior_focal_length) {
        continue;
      }
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const double error = CalculateAngularReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      if (error > max_angular_error_rad) {
        obs_to_delete.emplace_back(track_el.image_id, track_el.point2D_idx);
      }
    }
  }
  for (const auto& [image_id, point2D_idx] : obs_to_delete) {
    if (reconstruction_->Image(image_id).Point2D(point2D_idx).HasPoint3D()) {
      obs_manager.DeleteObservation(image_id, point2D_idx);
    }
  }

  // Filter tracks based on triangulation angle and reprojection error
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());
  // Set the threshold to be larger to avoid removing too many tracks
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10 * max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  // Normalize the structure for numerical stability.
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment(
    const BundleAdjustmentOptions& options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage,
    const std::function<bool()>& on_progress) {
  for (int ite = 0; ite < num_iterations; ite++) {
    // Optional fixed-rotation stage: optimize positions only
    if (!skip_fixed_rotation_stage) {
      BundleAdjustmentOptions opts_position_only = options;
      opts_position_only.constant_rig_from_world_rotation = true;
      // Caspar's pose node is a single retracted Pose3 (rotation+
      // translation together) with no mechanism to hold rotation
      // constant while translation is free -- constant_rig_from_world_
      // rotation is silently ignored when backend == CASPAR.
      if (opts_position_only.backend == BundleAdjustmentBackend::CASPAR) {
        opts_position_only.backend = BundleAdjustmentBackend::CERES;
      }
      if (!RunBundleAdjustment(opts_position_only, *reconstruction_)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << num_iterations << ", fixed-rotation stage finished";
    }

    // Joint optimization stage: default BA
    if (!skip_joint_optimization_stage) {
      if (!RunBundleAdjustment(options, *reconstruction_)) {
        return false;
      }
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << " finished";

    // Normalize the structure for numerical stability.
    // TODO: Skip normalization when position priors are used (similar to
    // incremental mapper's !use_prior_position condition).
    reconstruction_->Normalize();

    // Report progress for this refinement iteration and stop early if
    // requested. The filter passes above leave point3D.error in normalized
    // units, so recompute it in pixels first to keep intermediate
    // visualizations consistent with the final reconstruction.
    if (on_progress) {
      reconstruction_->UpdatePoint3DErrors();
      if (on_progress()) {
        break;
      }
    }

    // Filter tracks based on the estimation
    // For the filtering, in each round, the criteria for outlier is
    // tightened. If only few tracks are changed, no need to start bundle
    // adjustment right away. Instead, use a more strict criteria to filter
    LOG(INFO) << "Filtering tracks by reprojection ...";

    ObservationManager obs_manager(*reconstruction_);
    bool status = true;
    size_t filtered_num = 0;
    while (status && ite < num_iterations) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * max_normalized_reproj_error,
          reconstruction_->Point3DIds(),
          ReprojectionErrorType::NORMALIZED);

      if (filtered_num > 1e-3 * reconstruction_->NumPoints3D()) {
        status = false;
      } else {
        ite++;
      }
    }
    if (status) {
      LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
      break;
    }
  }

  // Filter tracks based on the estimation
  LOG(INFO) << "Filtering tracks by reprojection ...";
  {
    ObservationManager obs_manager(*reconstruction_);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        reconstruction_->Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        min_tri_angle_deg, reconstruction_->Point3DIds());
  }

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine(
    const IncrementalTriangulator::Options& options,
    const BundleAdjustmentOptions& ba_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();

  // Initialize mapper.
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  // Triangulate all registered images.
  for (const auto image_id : reconstruction_->RegImageIds()) {
    mapper.TriangulateImage(options, image_id);
  }

  // Set up bundle adjustment options for colmap's incremental mapper.
  BundleAdjustmentOptions custom_ba_options = ba_options;
  custom_ba_options.print_summary = false;
  if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
        ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_num_iterations = 50;
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations = 100;
  }

  // Iterative global refinement.
  IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options.random_seed;
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   custom_ba_options,
                                   options,
                                   /*normalize_reconstruction=*/true);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  ObservationManager obs_manager(*reconstruction_);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  if (!RunBundleAdjustment(ba_options, *reconstruction_)) {
    return false;
  }

  // Normalize the structure for numerical stability.
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());

  return true;
}

bool GlobalMapper::Solve(const GlobalMapperOptions& options,
                         const std::function<bool()>& on_progress) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Reports the current reconstruction and returns whether a stop was
  // requested. Point errors are recomputed in pixels before reporting because
  // the preceding filter passes leave point3D.error in normalized units, which
  // would otherwise make intermediate visualizations inconsistent with the
  // final reconstruction.
  const auto report_and_check_stop = [&]() {
    if (!on_progress) {
      return false;
    }
    reconstruction_->UpdatePoint3DErrors();
    return on_progress();
  };

  const bool use_pose_priors = !options.pose_prior_path.empty();
  if (use_pose_priors) {
    LoadPosePriors(options, *reconstruction_);
    const auto active_frames = pose_graph_->LargestConnectedFrameComponent(
        *reconstruction_, /*filter_unregistered=*/true);
    const auto reg_frame_ids = reconstruction_->RegFrameIds();
    for (const frame_t frame_id : reg_frame_ids) {
      if (!active_frames.count(frame_id)) {
        reconstruction_->DeRegisterFrame(frame_id);
      }
    }
    const auto image_ids = reconstruction_->RegImageIds();
    pose_graph_->InvalidatePairsOutsideActiveImageIds(
        {image_ids.begin(), image_ids.end()});
  }

  // Run rotation averaging
  if (!use_pose_priors && !options.skip_rotation_averaging) {
    LOG_HEADING1("Running rotation averaging");
    Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(options.RotationAveraging())) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Track establishment and selection
  if (!options.skip_track_establishment) {
    LOG_HEADING1("Running track establishment");
    Timer run_timer;
    run_timer.Start();
    EstablishTracks(options);
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Global positioning
  if (use_pose_priors) {
    LOG_HEADING1("Triangulating tracks from prior poses");
    TriangulateTracks(options, *reconstruction_);
    if (reconstruction_->NumPoints3D() == 0) {
      LOG(ERROR) << "Could not triangulate any tracks from the pose priors";
      return false;
    }
    if (report_and_check_stop()) {
      return true;
    }
  } else if (!options.skip_global_positioning) {
    LOG_HEADING1("Running global positioning");
    Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning(options.GlobalPositioning(),
                           options.max_angular_reproj_error_deg,
                           options.max_normalized_reproj_error,
                           options.min_tri_angle_deg)) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";

    // Report the first 3D view after global positioning and stop early if
    // requested.
    if (report_and_check_stop()) {
      return true;
    }
  }

  // Bundle adjustment
  if (!options.skip_bundle_adjustment) {
    LOG_HEADING1("Running iterative bundle adjustment");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment(options.BundleAdjustment(),
                                   options.max_normalized_reproj_error,
                                   options.min_tri_angle_deg,
                                   options.ba_num_iterations,
                                   options.ba_skip_fixed_rotation_stage,
                                   options.ba_skip_joint_optimization_stage,
                                   on_progress)) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Retriangulation
  if (!options.skip_retriangulation) {
    LOG_HEADING1("Running iterative retriangulation and refinement");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine(options.Retriangulation(),
                                         options.BundleAdjustment(),
                                         options.max_normalized_reproj_error,
                                         options.min_tri_angle_deg)) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";

    // Report the result after retriangulation and stop early if requested.
    if (report_and_check_stop()) {
      return true;
    }
  }

  // Filter passes here use NORMALIZED/ANGULAR error, so point3D.error is
  // left in non-pixel units. Recompute in pixels for consistent reporting
  // in model_analyzer.
  reconstruction_->UpdatePoint3DErrors();

  return true;
}

}  // namespace colmap
