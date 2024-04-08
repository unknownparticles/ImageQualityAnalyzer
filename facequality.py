import cv2
import dlib
import numpy as np

# 加载人脸检测器
detector = dlib.get_frontal_face_detector()
# 加载人脸关键点检测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 加载图像
img = cv2.imread('/Users/alun/Documents/20230804_智能挑图/模糊/134A1014.JPG')
# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用人脸检测器检测人脸
faces = detector(gray)

# 初始化变量
num_faces = len(faces)
face_data = []

# 遍历每个检测到的人脸
for i, face in enumerate(faces):
    # 获取人脸关键点
    landmarks = predictor(gray, face)
    
    # 初始化当前人脸的数据
    current_face_data = {
        'id': i + 1,
        'completeness': {},
        'blurriness': 0,
        'illumination': 0,
        'pose': {},
        'size': 0
    }
    
    # 将人脸框的坐标添加到当前人脸的数据中
    current_face_data['rect'] = {
        'left': face.left(),
        'top': face.top(),
        'right': face.right(),
        'bottom': face.bottom()
    }
    
    # 计算人脸完整度
    current_face_data['completeness']['full_face'] = 1 if face.left() > 0 and face.right() < gray.shape[1] and face.top() > 0 and face.bottom() < gray.shape[0] else 0
    current_face_data['completeness']['left_eye'] = 1 if landmarks.part(36).x > face.left() and landmarks.part(39).x < face.right() and landmarks.part(38).y > face.top() and landmarks.part(41).y < face.bottom() else 0
    current_face_data['completeness']['right_eye'] = 1 if landmarks.part(42).x > face.left() and landmarks.part(45).x < face.right() and landmarks.part(44).y > face.top() and landmarks.part(47).y < face.bottom() else 0
    current_face_data['completeness']['nose'] = 1 if landmarks.part(27).x > face.left() and landmarks.part(35).x < face.right() and landmarks.part(27).y > face.top() and landmarks.part(33).y < face.bottom() else 0
    current_face_data['completeness']['mouth'] = 1 if landmarks.part(48).x > face.left() and landmarks.part(54).x < face.right() and landmarks.part(50).y > face.top() and landmarks.part(57).y < face.bottom() else 0
    
    # 计算人脸模糊程度
    face_roi = gray[face.top():face.bottom(), face.left():face.right()]
    blurriness = cv2.Laplacian(face_roi, cv2.CV_64F).var()
    current_face_data['blurriness'] = min(blurriness / 1000, 1)
    
    # 计算光照范围
    current_face_data['illumination'] = np.mean(face_roi)
    
    # 计算人脸姿态角度
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # 鼻尖
        (landmarks.part(8).x, landmarks.part(8).y),    # 下巴
        (landmarks.part(36).x, landmarks.part(36).y),  # 左眼左角
        (landmarks.part(45).x, landmarks.part(45).y),  # 右眼右角
        (landmarks.part(48).x, landmarks.part(48).y),  # 左嘴角
        (landmarks.part(54).x, landmarks.part(54).y)   # 右嘴角
    ], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),             # 鼻尖
        (0.0, -330.0, -65.0),        # 下巴
        (-225.0, 170.0, -135.0),     # 左眼左角
        (225.0, 170.0, -135.0),      # 右眼右角
        (-150.0, -150.0, -125.0),    # 左嘴角
        (150.0, -150.0, -125.0)      # 右嘴角
    ])
    size = gray.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    current_face_data['pose']['pitch'] = euler_angles[0]
    current_face_data['pose']['yaw'] = euler_angles[1]
    current_face_data['pose']['roll'] = euler_angles[2]
    
    # 计算人脸大小
    face_size = (face.right() - face.left(), face.bottom() - face.top())
    current_face_data['size'] = 1 if face_size[0] > gray.shape[1] or face_size[1] > gray.shape[0] else f"{face_size[0]}px*{face_size[1]}px"
    
    # 将当前人脸的数据添加到总的人脸数据列表中
    face_data.append(current_face_data)

# 输出结果
print(f"人脸数量：{num_faces}")
for face in face_data:
    print(f"人脸{face['id']}：")
    print(f"  完整度：{face['completeness']}")
    print(f"  模糊程度：{face['blurriness']:.2f}")
    print(f"  光照范围：{face['illumination']:.2f}")
    print(f"  姿态角度：Pitch={face['pose']['pitch'][0]:.2f}, Yaw={face['pose']['yaw'][0]:.2f}, Roll={face['pose']['roll'][0]:.2f}")
    print(f"  大小：{face['size']}")

# 可选：输出人脸与人脸ID对应关系
for face_info in face_data:
    completeness = face_info['completeness']['full_face']
    left = completeness * face_info['rect']['left']
    top = completeness * face_info['rect']['top']
    right = completeness * face_info['rect']['right']
    bottom = completeness * face_info['rect']['bottom']
    
    cv2.putText(img, str(face_info['id']), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow("Faces with IDs", img)
cv2.waitKey(0)
cv2.destroyAllWindows()