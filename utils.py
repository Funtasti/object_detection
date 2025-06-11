import cv2

def extract_frames(video_path):
    """
    Generator : yields (frame_index, frame) tuples from input video.
    """
    video = cv2.VideoCapture(video_path)
    index = 0
    while video.isOpened():
        read, frame = video.read()
        if not read:
            break
        yield index, frame
        index += 1
    video.release()

def save_video(frames, fps, size, output_path):
    """
    Saves a list of frames as a video with given fps and frame size.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()