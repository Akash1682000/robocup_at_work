// Gazebo grasp plugin: joins the object between the fingers to the hand.
//
// ODE friction alone could not lift the objects, so /grasp/attach creates a
// real joint between panda_link7 and the object and /grasp/detach removes it.
// panda_link7 is used because gazebo merges the fixed hand links into it.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace robocup_grasp
{

class GraspPlugin : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model->GetWorld();
    ros_ = gazebo_ros::Node::Get(sdf);

    palm_name_ = read(sdf, "palm", "panda_link7");
    left_name_ = read(sdf, "left_finger", "panda_leftfinger");
    right_name_ = read(sdf, "right_finger", "panda_rightfinger");
    reach_ = sdf->HasElement("reach") ? sdf->Get<double>("reach") : 0.035;
    pad_ = sdf->HasElement("pad") ? sdf->Get<double>("pad") : 0.045;

    palm_ = model_->GetLink(palm_name_);
    left_ = model_->GetLink(left_name_);
    right_ = model_->GetLink(right_name_);

    if (!palm_ || !left_ || !right_) {
      RCLCPP_ERROR(ros_->get_logger(),
                   "grasp plugin: the model has no %s/%s/%s, doing nothing",
                   palm_name_.c_str(), left_name_.c_str(),
                   right_name_.c_str());
      return;
    }

    attach_srv_ = ros_->create_service<std_srvs::srv::Trigger>(
      "grasp/attach",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        (void)req;
        request(true, res);
      });

    detach_srv_ = ros_->create_service<std_srvs::srv::Trigger>(
      "grasp/detach",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        (void)req;
        request(false, res);
      });

    // joints are created/removed on the physics thread, not in the
    // service callback, to avoid touching the world mid-step
    tick_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      [this](const gazebo::common::UpdateInfo &) { pump(); });

    RCLCPP_INFO(ros_->get_logger(),
                "grasp plugin up: welding to %s, reach %.0f mm",
                palm_name_.c_str(), reach_ * 1000.0);
  }

private:
  static std::string read(sdf::ElementPtr sdf, const std::string & name,
                          const std::string & fallback)
  {
    return sdf->HasElement(name) ? sdf->Get<std::string>(name) : fallback;
  }

  // queue the request for the physics thread and wait for the result
  void request(bool attach,
               std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    want_ = attach ? Want::Attach : Want::Detach;
    done_ = false;
    if (!cv_.wait_for(lock, std::chrono::seconds(5),
                      [this] { return done_; })) {
      res->success = false;
      res->message = "the physics thread never picked the request up";
      return;
    }
    res->success = ok_;
    res->message = said_;
  }

  void pump()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (want_ == Want::Nothing) {
      return;
    }
    if (want_ == Want::Attach) {
      ok_ = attach();
    } else {
      ok_ = detach();
    }
    want_ = Want::Nothing;
    done_ = true;
    lock.unlock();
    cv_.notify_all();
  }

  // Point between the fingertips. The finger link origins are ~45 mm
  // behind the tips, so they would need a much larger reach.
  ignition::math::Vector3d grip() const
  {
    const ignition::math::Pose3d l = left_->WorldPose();
    const ignition::math::Pose3d r = right_->WorldPose();
    const ignition::math::Vector3d down =
      l.Rot().RotateVector(ignition::math::Vector3d::UnitZ);
    return (l.Pos() + r.Pos()) / 2.0 + down * pad_;
  }

  // Nearest non-static model within reach (skips the robot itself).
  gazebo::physics::LinkPtr candidate(std::string & name) const
  {
    const ignition::math::Vector3d at = grip();
    gazebo::physics::LinkPtr best;
    double best_d = reach_;

    for (const auto & other : world_->Models()) {
      if (!other || other == model_ || other->IsStatic()) {
        continue;
      }
      for (const auto & link : other->GetLinks()) {
        if (!link) {
          continue;
        }
        const double d = (link->WorldPose().Pos() - at).Length();
        if (d < best_d) {
          best_d = d;
          best = link;
          name = other->GetName();
        }
      }
    }
    return best;
  }

  bool attach()
  {
    if (joint_) {
      said_ = "already holding " + held_;
      return false;
    }

    std::string name;
    gazebo::physics::LinkPtr prize = candidate(name);
    if (!prize) {
      said_ = "nothing within " + std::to_string(int(reach_ * 1000.0)) +
              " mm of the fingers";
      return false;
    }

    // revolute with zero limits, the fixed joint drifts under load
    joint_ = world_->Physics()->CreateJoint("revolute", model_);
    joint_->Attach(palm_, prize);
    joint_->Load(palm_, prize, ignition::math::Pose3d());
    joint_->SetModel(model_);
    joint_->SetUpperLimit(0, 0.0);
    joint_->SetLowerLimit(0, 0.0);
    joint_->Init();

    held_ = name;
    said_ = "holding " + name;
    RCLCPP_INFO(ros_->get_logger(), "grasp: welded %s to the palm",
                name.c_str());
    return true;
  }

  bool detach()
  {
    if (!joint_) {
      said_ = "not holding anything";
      return false;
    }
    joint_->Detach();
    joint_.reset();
    said_ = "released " + held_;
    RCLCPP_INFO(ros_->get_logger(), "grasp: released %s", held_.c_str());
    held_.clear();
    return true;
  }

  enum class Want { Nothing, Attach, Detach };

  gazebo::physics::ModelPtr model_;
  gazebo::physics::WorldPtr world_;
  gazebo::physics::LinkPtr palm_, left_, right_;
  gazebo::physics::JointPtr joint_;
  gazebo::event::ConnectionPtr tick_;

  gazebo_ros::Node::SharedPtr ros_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr attach_srv_, detach_srv_;

  std::string palm_name_, left_name_, right_name_, held_, said_;
  double reach_{0.035};
  double pad_{0.045};

  std::mutex mutex_;
  std::condition_variable cv_;
  Want want_{Want::Nothing};
  bool done_{false};
  bool ok_{false};
};

GZ_REGISTER_MODEL_PLUGIN(GraspPlugin)

}  // namespace robocup_grasp
