#!/usr/bin/env python3


import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data, QoSReliabilityPolicy, QoSDurabilityPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped, PointStamped, TransformStamped, Point, Pose, Quaternion
from visualization_msgs.msg import MarkerArray

from scipy.spatial.transform import Rotation as R
from scipy.stats import linregress

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO
import pyransac3d as pyrsc

import tf2_ros
import tf2_geometry_msgs

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

from time import time as tik
       

class detect_faces(Node):

    def __init__(self, frequency=5):
        super().__init__('detect_faces')
        self.frequency = frequency
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
        ])

        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.timer_period = 1/frequency
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        client_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=timer_callback_group)

        self.device = self.get_parameter('device').get_parameter_value().string_value
        max_face_dist_param = self.declare_parameter('max_face_dist', 1.)
        running_mean_window_size_param = self.declare_parameter('running_mean_window_size', 10)
        normal_sim_tol_param = self.declare_parameter('normal_sim_tol', 1e-2)
        
        self.bridge = CvBridge()
        self.scan = None
        marker_topic = "/task_1/people_markers"
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data,
                                                       callback_group=client_callback_group )
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
        self.image_pub  = self.create_publisher(Image, "/image", QoSReliabilityPolicy.BEST_EFFORT)
        self.people_cloud_pub  = self.create_publisher(PointCloud2, "/task_1/face_points", qos_profile)

        self.model = YOLO("yolov8n.pt")

        self.faces = []

        self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")
        self.tf_buffer = tf2_ros.Buffer(rclpy.time.Duration(seconds=15.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()


    def make_marker(self, face, i):
        scale = 0.05
        (x, y, z), (a, b, c) = face
        o, p, q, r = R.from_euler('xyz', [0, 0, -np.arctan2(b, a)]).as_quat().astype(float)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.lifetime = rclpy.time.Duration(seconds=2/self.frequency).to_msg()
        marker.type = 0
        marker.id = i
        marker.scale.x = scale * 4
        marker.scale.y = scale * 1
        marker.scale.z = scale * 1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        t = Point()
        t.x = x
        t.y = y
        t.z = z
        n = Quaternion()
        n.x = o
        n.y = p
        n.z = q
        n.w = r
        marker.pose = Pose(position=t, orientation=n)
        return marker


    def timer_callback(self):
        i = 0
        marker_array = MarkerArray()
        pc_points = []
        pc_colors = []
        for face in self.faces:
            centers, normals, points, colors = face
            if centers.shape[0] > 3:
                center = centers.mean(axis=0)
                normal = normals.mean(axis=0)
                marker = self.make_marker((center, normal), i)
                i += 1
                marker_array.markers.append(marker)
                pc_points.append(points)
                pc_colors.append(colors)
        self.marker_pub.publish(marker_array)
        if len(pc_points) > 0:
            pc_points = np.vstack(pc_points)
            pc_colors = np.vstack(pc_colors)
            header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
            pc2 = self.point_cloud(header, pc_points, pc_colors)
            self.people_cloud_pub.publish(pc2)
        #print(i)


    def add_face(self, face):
        max_face_dist = self.get_parameter('max_face_dist').get_parameter_value().double_value
        normal_sim_tol = self.get_parameter('normal_sim_tol').get_parameter_value().double_value

        running_mean_window_size = self.get_parameter('running_mean_window_size').get_parameter_value().integer_value
        candidate_center, candidate_normal, points, colors = face
        if np.any(np.isnan(candidate_center)):
            return
        for filtered_face in self.faces:
            compare_centers, compare_normals = filtered_face[:2]
            cos_sim = np.dot(compare_normals.mean(axis=0), candidate_normal)
            same_normal = np.isclose(cos_sim, 1, atol=normal_sim_tol)
            same_center = np.linalg.norm(compare_centers.mean(axis=0) - candidate_center) < max_face_dist
            #print("same_center", same_center, "same_normal", same_normal)
            if not same_normal:
                pass
                #print(candidate_normal, compare_normals.mean(axis=0), np.dot(compare_normals.mean(axis=0), candidate_normal))
            if same_normal and same_center:
                clip = 1 if compare_centers.shape[0] > running_mean_window_size else 0
                filtered_face[0] = np.vstack((compare_centers[clip:], candidate_center))
                filtered_face[1] = np.vstack((compare_normals[clip:], candidate_normal))
                if points.size > filtered_face[2].size:
                        filtered_face[2] = points
                        filtered_face[3] = colors
                return
            elif same_center:
                opposite_normal = np.isclose(cos_sim, -1, atol=normal_sim_tol)
                if not opposite_normal:
                    return
        filtered_face = [np.array([candidate_center]), np.array([candidate_normal]), points, colors]
        self.faces.append(filtered_face)
                
            
    def pointcloud_callback(self, pointcloud_message):
        header = pointcloud_message.header
        try:
            depth_to_world_transform : TransformStamped = self.tf_buffer.lookup_transform(
                    'map',
                    header.frame_id, 
                    self.tf_buffer.get_latest_common_time('map', header.frame_id))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e: 
            print(e)
            return None
        
        tok = tik()
        # process pointclod message
        h, w = pointcloud_message.height, pointcloud_message.width
        points = pc2.read_points_numpy(pointcloud_message, field_names=("x", "y", "z")).reshape(h, w, 3)
        rgb = pc2.read_points_numpy(pointcloud_message, field_names=("rgb",))
        rgb = rgb.view(np.uint8).reshape(h, w, 4)[..., :3]
        rgb = np.pad(rgb,((8,8), (0,0), (0,0)))
        depth = np.linalg.norm(points, axis=-1)
        
        #img = self.bridge.cv2_to_imgmsg(rgb)
        #self.image_pub.publish(img)

        # run inference
        res = self.model.predict(rgb, imgsz=rgb.shape[:2], show=False, verbose=False, classes=[0], device=self.device)

        # iterate over results
        for detection in res:
            bbox = detection.boxes.xyxy
            if bbox.nelement() == 0: # skip if empty
                continue

            self.get_logger().info(f"Person has been detected!")

            face = self.process_person_detection(detection, depth_to_world_transform.transform, points, rgb)
            self.add_face(face)


    def process_person_detection(self, detection, transform, points, rgb):
        
        translation = transform.translation
        trans_vec = [translation.x, translation.y, translation.z]
        rotation = transform.rotation
        rot = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])

        h, w = points.shape[:2]     
        bbox = detection.boxes.xyxy[0]    
        x1, y1, x2, y2 = [int(t) for t in bbox.cpu().numpy()]

        det_h, det_w = int(x2 - x1), int(y2 - y1)
        points_t = points[y1:y2, x1:x2]
        rgb_det = rgb[y1:y2, x1:x2].reshape(-1, 3)
        points_shape = points_t.shape
        points_t = rot.apply(points_t.reshape(-1, 3))#.reshape(points_shape)
        points_t = points_t + trans_vec
        '''normal = np.arctan2((points[h//2,x1-5:x1+5,1].mean() - points[h//2,x2-5:x2+5,1].mean()),
                             (points[h//2,x1-5:x1+5,0].mean() - points[h//2,x2-5:x2+5,0].mean()))'''
        plane1 = pyrsc.Plane()
        model, inliers = plane1.fit(points_t, thresh=.005, minPoints=10, maxIteration=50)
        center = points_t.mean(axis=(0,))
        robot_pos = rot.apply([0,0,0]) + trans_vec
        normal = model[:3]/np.linalg.norm(model[:3])
        axis = np.argmax(np.abs(normal))
        dir = 1 if axis == 1 else -1
        dir = -dir if robot_pos[axis] < center[axis] else dir
        normal = np.abs(normal) * dir

        return center, normal, points_t[inliers], rgb_det[inliers].astype(np.uint8)


    def point_cloud(self, header, points, colors):
        fields = [
            PointField(offset=0, name='x', count=1, datatype=PointField.FLOAT32),
            PointField(offset=4, name='y', count=1, datatype=PointField.FLOAT32),
            PointField(offset=8, name='z', count=1, datatype=PointField.FLOAT32),
            PointField(offset=12, name='rgb', count=1, datatype=PointField.UINT32)]
        
        data = np.hstack([points.astype(np.float32).view(np.uint8), colors.view(np.uint8)])

        return PointCloud2(
            header=header,
            height=1,
            width=data.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=data.shape[1],
            row_step=data.size,
            data=data.tobytes()
        )


def main():
    print('Face detection node starting.')

    rclpy.init(args=None)
    node = detect_faces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
