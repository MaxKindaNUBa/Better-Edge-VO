import cv2
import numpy as np

def generate_fixed_hybrid_videos():
    rgb_video = 'sample_rgbd/walking_xyz_rgb.avi'
    depth_video = 'sample_rgbd/walking_xyz_depth.avi'
    
    cap_rgb = cv2.VideoCapture(rgb_video)
    cap_d = cv2.VideoCapture(depth_video)
    
    total_rgb_frames = cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT)
    total_d_frames = cap_d.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # 1. Temporal Sync: Match depth playback speed to RGB playback speed
    sync_ratio = total_d_frames / total_rgb_frames
    
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    w = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_frames = 570 # 19.0 seconds at 30 fps
    
    temp_overlay_avi = '/tmp/overlay_fixed.avi'
    temp_sbs_avi = '/tmp/sbs_fixed.avi'
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_overlay = cv2.VideoWriter(temp_overlay_avi, fourcc, fps, (w, h))
    out_sbs = cv2.VideoWriter(temp_sbs_avi, fourcc, fps, (w * 2, h))
    
    # Kernel for cross-sensor baseline disparity tolerance
    d_gate_kernel = np.ones((7, 7), np.uint8)
    
    print(f"Generating fixed videos ({total_frames} frames)...")
    
    for i in range(total_frames):
        # Precise temporal sync per frame
        d_frame_idx = int(i * sync_ratio)
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_d.set(cv2.CAP_PROP_POS_FRAMES, d_frame_idx)
        
        ret_rgb, f_rgb = cap_rgb.read()
        ret_d, f_d = cap_d.read()
        if not ret_rgb or not ret_d:
            break
            
        gray = cv2.cvtColor(f_rgb, cv2.COLOR_BGR2GRAY)
        d_gray = cv2.cvtColor(f_d, cv2.COLOR_BGR2GRAY)
        
        # 1. Photometric Canny on RGB (guaranteed 100% pixel-perfect coordinates)
        rgb_blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
        rgb_edges = cv2.Canny(rgb_blur, 35, 110)
        
        # 2. Geometric Discontinuity on Depth
        d_blur = cv2.GaussianBlur(d_gray, (5, 5), 1.2)
        d_edges = cv2.Canny(d_blur, 20, 60)
        
        # 3. Cross-Bilateral Geometric Gating
        d_dilated = cv2.dilate(d_edges, d_gate_kernel)
        
        # Confirmed 3D structural edges (RGB edges verified by physical depth step/crease)
        structural_edges = cv2.bitwise_and(rgb_edges, d_dilated)
        
        # Flat surface texture edges (posters, floor patterns, shadows)
        texture_edges = cv2.bitwise_and(rgb_edges, cv2.bitwise_not(d_dilated))
        
        # 4. Construct Overlay Frame
        overlay_frame = f_rgb.copy()
        
        # Draw confirmed 3D Structural Boundaries in high-contrast Neon Green
        overlay_frame[structural_edges > 0] = [0, 255, 0]
        
        # Draw Flat Texture Edges in dim subtle Amber (so user can distinguish geometry vs texture)
        overlay_frame[texture_edges > 0] = [0, 140, 255]
        
        # Semi-transparent HUD Header Bar
        cv2.rectangle(overlay_frame, (0, 0), (w, 36), (20, 20, 20), -1)
        time_sec = i / fps
        cv2.putText(overlay_frame, "HYBRID RGB-D GATED EDGES | GREEN: 3D Geometry | ORANGE: Texture", (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay_frame, f"t={time_sec:.1f}s", (w - 70, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        out_overlay.write(overlay_frame)
        
        # 5. Side-by-Side Frame
        sbs_frame = np.hstack([f_rgb, overlay_frame])
        cv2.rectangle(sbs_frame, (0, 0), (w*2, 36), (15, 15, 15), -1)
        cv2.putText(sbs_frame, "ORIGINAL CAMERA FEED (TUM walking_xyz)", (15, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(sbs_frame, "HYBRID RGB-D EDGE OVERLAY (PIXEL-LOCKED)", (w + 15, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(sbs_frame, f"{time_sec:.1f}s", (w*2 - 75, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        
        out_sbs.write(sbs_frame)
        
        if (i + 1) % 100 == 0:
            print(f"  Rendered {i+1}/{total_frames} frames...")

    cap_rgb.release()
    cap_d.release()
    out_overlay.release()
    out_sbs.release()
    print("Frames rendered successfully!")

generate_fixed_hybrid_videos()
