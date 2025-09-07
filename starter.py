import torch
import torch.nn as nn
from collections import deque

class FrameFeatureExtractor(nn.Module):
    # small 2D CNN for per-frame feature extraction
    def __init__(self, output_size=256):
        super().__init__()
        self.features = nn.Sequential( # e.g., a few conv+pool layers
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        # x is a single frame [batch=1, 3, H, W]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TemporalModel(nn.Module):
    # e.g., a 1D CNN for processing the sequence of feature vectors
    def __init__(self, input_size=256, num_features=100, seq_length=3000):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, num_features, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1) # Global Avg Pool over time
        )
    def forward(self, x):
        # x shape: [batch, seq_length, input_size]
        x = x.transpose(1, 2) # -> [batch, input_size, seq_length] for Conv1d
        return self.tcn(x).squeeze() # -> [batch, num_features]

# Initialize
frame_cnn = FrameFeatureExtractor().to(device).eval()  # we don't train this initially
temporal_model = TemporalModel().to(device)
projection_net = ProjectionNet().to(device)

# Fixed-size buffer to hold feature vectors
sequence_buffer = deque(maxlen=3000)  # 3 seconds of data at 1000 fps

# Main Acquisition Loop
cap = get_1000fps_camera()

try:
    while True:
        # 1. Capture single frame
        ret, frame_t = cap.read()
        if not ret:
            break

        # 2. Preprocess frame: crop, normalize, convert to tensor
        frame_tensor = preprocess_frame(frame_t).unsqueeze(0).to(device) # shape [1, C, H, W]

        # 3. EXTRACT FEATURES AND FREE FRAME
        with torch.no_grad(): # No grad for efficiency during capture
            feature_vector_t = frame_cnn(frame_tensor) # shape [1, 256]
        feature_vector_t_cpu = feature_vector_t.squeeze().cpu().numpy()

        # 4. Store feature vector, discard old ones automatically
        sequence_buffer.append(feature_vector_t_cpu)

        # 5. Check if it's time to process (e.g., user stopped speaking)
        if user_finished_speaking_event.is_set():
            # Convert buffer to tensor for temporal model
            sequence_tensor = torch.tensor(np.array(sequence_buffer)).unsqueeze(0).to(device) # [1, 3000, 256]

            # Now run temporal model, projection, and training
            P = temporal_model(sequence_tensor)
            P_proj = projection_net(P)
            # ... [Rest of training code] ...

finally:
    cap.release()