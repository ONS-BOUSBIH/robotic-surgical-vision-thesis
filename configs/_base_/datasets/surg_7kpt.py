# Metadata file fpr the SurgPose dataset
dataset_info = dict(
    dataset_name='surgpose_7kpt',
    paper_info=dict(
        author='Wu, Zijian and Schmidt, Adam and Moore, Randy and Zhou, Haoying '
               'and Banks, Alexandre and Kazanzides, Peter and Salcudean, Septimiu E.',
        title='SurgPose: a Dataset for Articulated Robotic Surgical Tool Pose Estimation and Tracking',
        container='IEEE International Conference on Robotics and Automation (ICRA)',
        year='2025',
        homepage='https://surgpose.github.io/',
    ),
   keypoint_info={
        0: dict(name='shaft', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='wrist_pivot_1', id=1, color=[0, 255, 0], type='', swap=''),
        2: dict(name='wrist_pivot_2', id=2, color=[0, 0, 255], type='', swap=''),
        3: dict(name='clasper_tip_1', id=3, color=[255, 255, 0], type='', swap=''),
        4: dict(name='clasper_tip_2', id=4, color=[255, 0, 255], type='', swap=''),
        5: dict(name='redundant_wrist_mark', id=5, color=[0, 255, 255], type='', swap=''),
        6: dict(name='redundant_housing_mark', id=6, color=[128, 128, 128], type='', swap=''),
    },
    skeleton_info={
        0: dict(link=('shaft', 'wrist_pivot_1'), id=0, color=[100, 150, 200]),
        1: dict(link=('wrist_pivot_1', 'wrist_pivot_2'), id=1, color=[100, 150, 200]),
        2: dict(link=('wrist_pivot_2', 'clasper_tip_1'), id=2, color=[100, 150, 200]),
        3: dict(link=('wrist_pivot_2', 'clasper_tip_2'), id=3, color=[100, 150, 200]),
    },
    joint_weights=[1.0] * 7,
    # First 7 pf COCO sigmas for OKS computation
    sigmas=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079])

