from src.Segmentation.inference.SAM2_inferencer import SAM2SegmentationInferencer
import yaml
import argparse


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the inference config file')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        config_data = yaml.safe_load(f)

    source_images = config_data['images']
    source_bboxes = config_data['annotations']
    output_dir = config_data['output']
    split_file= config_data['split_file']
    
    with open(split_file, 'r') as f: 
        splits= yaml.safe_load(f) 
    vids= splits[config_data['split_value']]
    
    processor = SAM2SegmentationInferencer (model_path= config_data['model'], device="cuda", area_threshold=config_data['area_threshold'], w_threshold=config_data['width_threshold'])
    processor.run_inference(source_images, source_bboxes, output_dir, vids)


if __name__=="__main__":
    main()
