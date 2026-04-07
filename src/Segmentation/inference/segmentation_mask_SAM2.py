from src.Segmentation.inference.SAM2_inferencer import SAM2SegmentationInferencer
import yaml

def main():
    source_images = "data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/extracted_frames"
    source_bboxes = "data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/extracted_bboxes_kpts"
    output_dir = "data/Surgpose_for_segmentation"
    split_file="data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/video_split.yaml"
    
    with open(split_file, 'r') as f: 
        splits= yaml.safe_load(f) 
    vids= splits['test']
    processor = SAM2SegmentationInferencer (device="cuda")
    processor.run_inference(source_images, source_bboxes, output_dir, vids)


if __name__=="__main__":
    main()
