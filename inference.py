import os
import torch
import cv2 as cv
from video_dataset import VideoDataset
from model import generate_model
from opts import parse_opts
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
def read_video(video_path, transform, num_frames):
    '''
    Function to read and preprocess video
    '''
    frames = []
    cap = cv.VideoCapture(video_path)
    count_frames = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if transform:
                transformed = transform(image=frame)
                frame = transformed['image']
            frames.append(frame)
            count_frames += 1
        else:
            break

    stride = count_frames // num_frames
    new_frames = []
    count = 0
    for i in range(0, count_frames, stride):
        if count >= num_frames:
            break
        new_frames.append(frames[i])
        count += 1

    cap.release()
    return torch.stack(new_frames, dim=0)

def infer_video(video_path, model, device, transform, num_frames):
    '''
    Perform inference on a single video
    '''
    frames = read_video(video_path, transform, num_frames)
    frames = frames.unsqueeze(0)  # Add batch dimension
    frames = frames.to(device)

    with torch.no_grad():
        outputs = model(frames)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.item()


    
if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    device = torch.device("cpu")
    model = generate_model(opt, device)
    
    load_model(opt.resume_path, model)
    model.eval()
    
    transform = A.Compose([
        A.Resize(height=opt.sample_size, width=opt.sample_size),  # Adjust the size as per your model
        A.Normalize(),
        ToTensorV2()
    ])
    
    video_path = opt.predict_video_path
    prediction = infer_video(video_path, model, device, transform, opt.sample_duration)
    
    type_dict = [
        "Biking",
        "CliffDiving",
        "Drumming",
        "Haircut",
        "HandstandWalking",
        "HighJump",
        "JumpingJack",
        "Mixing",
        "Skiing",
        "SumoWrestling"
    ]
    
    # make sure prediction is in [0, 9]
    prediction = max(0, prediction)
    prediction = min(9, prediction)
    
    print(f"Predicted class for the video: {type_dict[prediction]}")