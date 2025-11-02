import mediapipe as mp

def get_blank_landmark(num_points):
        return [[None, None, None] for _ in range(num_points)]

class BaseLandmaker():
    def __init__(self, model_path, num_objects=1):
        self.model_path = model_path
        self.num_objects = num_objects
        self.running_mode = mp.tasks.vision.RunningMode
        self.detector = None

    def create_task(self):
        """To be implement by child classes"""
        raise NotImplemented("Subclass must implement this method")
    
    def close_task(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
            print("Landmarker closed!")

    # def blank_landmark(self, num_points):
    #     return get_blank_landmark(num_points)

    # def process(self, frame, frame_timestamp_ms, is_used=True):
    #     if is_used:
    #         return self.process(frame, frame_timestamp_ms)
    #     else:
    #         return self.blank_landmark()

class FaceLandmarker(BaseLandmaker):
    def __init__(self, model_path='models/face_landmarker.task', num_objects=1):
        super().__init__(model_path=model_path, num_objects=num_objects)

    def create_task(self, mode='video'):
        self.task_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path),
            num_faces = self.num_objects,
            running_mode = self.running_mode.VIDEO if mode == 'video' else self.running_mode.IMAGE
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(self.task_options)
        print("Face Landmarker created!")

    def process_frame(self, frame, frame_timestamp_ms):
        """
        Output:
        """
        face = []
        result = self.detector.detect_for_video(frame, frame_timestamp_ms)

        # Get key points' coordinates
        face = [[key_point.x, key_point.y, key_point.z] for key_point in result.face_landmarks[0]]
        
        # Fill in with [None, None, None] key points if landmarks are not detected
        if not face:
            face = get_blank_landmark(468)

        return face
    
    def get_empty_face(self):
        return get_blank_landmark(468)

class HandsLandmarker(BaseLandmaker):
    def __init__(self, model_path='models/hand_landmarker.task',  num_objects=2):
        super().__init__(model_path=model_path, num_objects=num_objects)
        
    def create_task(self, mode='video'):
        self.task_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path),
            num_hands = self.num_objects,
            running_mode = self.running_mode.VIDEO if mode == 'video' else self.running_mode.IMAGE
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(self.task_options)
        print("Hands Landmarker created!")

    def process_frame(self, frame, frame_timestamp_ms):
        """
        Output:
            Trả về list hoặc dict các 42 điểm key points của cả 2 tay.
        """
        left_hand = []
        right_hand = []
        result = self.detector.detect_for_video(frame, frame_timestamp_ms)

        # Get key points' coordinates
        for i in range(len(result.handedness)):
            if result.handedness[i][0].category_name == 'Left':
                left_hand = [[key_point.x, key_point.y, key_point.z] for key_point in result.hand_landmarks[i]]
            else:
                right_hand = [[key_point.x, key_point.y, key_point.z] for key_point in result.hand_landmarks[i]]

        # Fill in with [None, None, None] key points if landmarks are not detected
        if not left_hand:
            left_hand = get_blank_landmark(21)
        
        if not right_hand:
            right_hand = get_blank_landmark(21)

        return left_hand + right_hand
    
    def get_empty_hands(self):
        return get_blank_landmark(42)

class PoseLandmarker(BaseLandmaker):
    def __init__(self, model_path='models/pose_landmarker_lite.task', num_objects=2):
        super().__init__(model_path=model_path, num_objects=num_objects)

    def create_task(self, mode='video'):
        self.task_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path),
            num_poses = self.num_objects,
            running_mode = self.running_mode.VIDEO if mode == 'video' else self.running_mode.IMAGE
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(self.task_options)
        print("Pose Landmarker Created!")

    def process_frame(self, frame, frame_timestamp_ms):
        """
        Output:
        """
        pose = []
        result = self.detector.detect_for_video(frame, frame_timestamp_ms)

        # Get key points' coordinates
        pose = [[key_point.x, key_point.y, key_point.z] for key_point in result.pose_landmarks[0]]
        
        # Fill in with [None, None, None] key points if landmarks are not detected
        if not pose:
            pose = get_blank_landmark(33)

        return pose
    
    def get_empty_pose(self):
        return get_blank_landmark(33)