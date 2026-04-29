from src.Stereo_matching.inference.S2M2_inferencer import S2M2Inferencer
import yaml
import argparse


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the inference config file')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        config_data = yaml.safe_load(f)

    left_folder = config_data['left_img_root']
    right_folder = config_data['right_img_root']
    zip_root= config_data['zip_root']
    output_dir = config_data['output']
    split_file= config_data['split_file']
    
    with open(split_file, 'r') as f: 
        splits= yaml.safe_load(f) 
    vids= splits[config_data['split_value']]

    inferencer = S2M2Inferencer(config_data['pretrained_folder'], model_type=config_data['model_type'],num_refine= config_data['num_refine'], allow_negative= config_data['allow_negative'],device='cuda')
    inferencer.run_batch_inference(left_img_root=left_folder, right_img_root=right_folder, zip_root=zip_root,output_dir=output_dir, video_ids=vids, img_shape= (config_data['h'],config_data['w']), lrc_threshold= config_data['lrc_threshold'],save_visuals=config_data['save_visuals'],resized=config_data['resized'])

if __name__=="__main__":
    main()