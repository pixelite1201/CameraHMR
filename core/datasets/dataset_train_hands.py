import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from ..smplx_wrapper import SMPLX_
from ..utils.pylogger import get_pylogger
from ..configs import DATASET_FOLDERS, DATASET_FILES
from ..constants import FLIP_KEYPOINT_PERMUTATION, NUM_BETAS_SMPLX, SMPLX_MODEL_DIR
from .utils import resize_image
from .utils_hands import expand_to_aspect_ratio, get_example
from torchvision.transforms import Normalize

log = get_pylogger(__name__)
np2th = lambda x: torch.from_numpy(x).float()

class DatasetTrain(Dataset):
    def __init__(self, cfg, dataset, is_train=True):
        super(DatasetTrain, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.cfg = cfg
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.normalize_img = Normalize(mean=cfg.MODEL.IMAGE_MEAN, std=cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.img_dir = DATASET_FOLDERS[dataset]
        self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        self.imgname = self.data['imgname']
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.center_lh = self.data['lh_c']
        self.center_rh = self.data['rh_c']
        self.scale_lh = self.data['lh_s']
        self.scale_rh = self.data['rh_s']

        self.pose = self.data['pose_cam'].astype(np.float32)
        if self.data['shape'].shape[1] <NUM_BETAS_SMPLX:
            missing_betas =NUM_BETAS_SMPLX - self.data['shape'].shape[1]
            self.betas = self.data['shape'].astype(np.float32)
            zeros = np.zeros((self.data['shape'].shape[0], missing_betas), dtype=np.float32)
            self.betas = np.concatenate([self.betas, zeros], axis=1)
        else:
            self.betas = self.data['shape'].astype(np.float32)[:, :NUM_BETAS_SMPLX]
        self.cam_int = self.data['cam_int']
        self.keypoints = self.data['gtkps']
        self.length = self.scale.shape[0]
        if 'cam_ext' in self.data:
            self.cam_ext = self.data['cam_ext']
        else:
            self.cam_ext = np.zeros((self.length, 4, 4))

        self.trans_cam = self.data['trans_cam']

        lh_det_confs = np.ones((self.data['shape'].shape[0], 21 ,1))
        rh_det_confs = np.ones((self.data['shape'].shape[0], 21, 1))
        self.lh_v_conf =  np2th(np.ones((self.data['shape'].shape[0], 778, 1)))
        self.rh_v_conf = np2th(np.ones((self.data['shape'].shape[0], 778, 1)))
        self.hand_conf = np2th(np.concatenate([lh_det_confs, rh_det_confs], axis=1) )
        self.joints3d_conf = np.ones((171,1))
        self.smplx_gt = SMPLX_(model_path=SMPLX_MODEL_DIR, use_pca=False, num_betas=NUM_BETAS_SMPLX, flat_hand_mean=True)

        log.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints_2d = self.keypoints[index].copy()
        orig_keypoints_2d = self.keypoints[index].copy()

        center_x, center_y = center[0], center[1]
        center_lh, center_rh = self.center_lh[index], self.center_rh[index]
        center_lx, center_ly = center_lh[0], center_lh[1]
        center_rx, center_ry = center_rh[0], center_rh[1]
        scale_lh, scale_rh = self.scale_lh[index], self.scale_rh[index]

        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=self.BBOX_SHAPE).max()
        lh_bbox_size = expand_to_aspect_ratio(scale_lh * 200, target_aspect_ratio=self.BBOX_SHAPE).max()
        rh_bbox_size = expand_to_aspect_ratio(scale_rh * 200, target_aspect_ratio=self.BBOX_SHAPE).max()

        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        imgname = os.path.join(self.img_dir, self.imgname[index])
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if cv_img is None:
            log.error(f"Failed to read image: {imgname}")
            return self.__getitem__((index + 1) % self.length)
        cv_img = cv_img[:, :, ::-1]

        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized_norm = np.transpose(img_full_resized.astype('float32'), (2, 0, 1)) / 255.0
        item['img_full_resized'] = self.normalize_img(torch.from_numpy(img_full_resized_norm).float())

        smplx_params = {
            'global_orient': self.pose[index][:3].astype(np.float32),
            'body_pose': self.pose[index][3:66].astype(np.float32),
            'left_hand_pose': self.pose[index][75:120].astype(np.float32),
            'right_hand_pose': self.pose[index][120:165].astype(np.float32),
            'jaw_pose': np.zeros(3, dtype=np.float32),
            'leye_pose': np.zeros(3, dtype=np.float32),
            'reye_pose': np.zeros(3, dtype=np.float32),
            'expression': np.zeros(10, dtype=np.float32),
            'betas': self.betas[index].astype(np.float32)
        }
        item['smpl_params'] = smplx_params

        smplx_output_gt = self.smplx_gt(**{k: np2th(v).unsqueeze(0) for k, v in smplx_params.items()})
        item['keypoints_3d'] = torch.cat((smplx_output_gt.joints.detach()[0], np2th(self.joints3d_conf)), dim=-1)
        item['gt_hand_joints'] = torch.cat([smplx_output_gt.hand_joints[0].detach(), self.hand_conf[index]], dim=-1).float()
        item['gt_lhand_verts'] =  torch.cat([smplx_output_gt.lhand_verts[0].detach(), self.lh_v_conf[index]], dim=-1).float()
        item['gt_rhand_verts'] =  torch.cat([smplx_output_gt.rhand_verts[0].detach(), self.rh_v_conf[index]], dim=-1).float()
        item['gt_feet_joints'] = smplx_output_gt.feet_joints[0].detach()
        item['gt_face_joints'] = smplx_output_gt.face_joints[0].detach()
        item['gt_body_verts'] = smplx_output_gt.body_verts[0].detach()
        item['gt_vertices'] = smplx_output_gt.vertices[0].detach()

        translation = self.cam_ext[index][:, 3].copy()
        if 'trans_cam' in self.data.files:
            translation[:3] += self.trans_cam[index]
        item['translation'] = translation

        img_patch_rgba, \
        img_patch_cv, \
        keypoints_2d, \
        img_size, cx, cy, trans_lh, trans_rh, bbox_w, bbox_h, trans, scale_aug = get_example(
            imgname,
            center_x, center_y, center_lx, center_ly, center_rx, center_ry,
            bbox_size, bbox_size, lh_bbox_size, rh_bbox_size,
            keypoints_2d,
            FLIP_KEYPOINT_PERMUTATION,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=True, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode,
            dataset=self.dataset
        )

        item['img'] = img_patch_rgba[:3, :, :]
        item['img_disp'] = img_patch_cv
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = np.array([cx, cy])
        item['box_size'] = bbox_w * scale_aug
        item['img_size'] = 1.0 * img_size.copy()
        item['cam_int'] = np.array(self.cam_int[index]).astype(np.float32)
        item['imgname'] = imgname
        item['dataset'] = self.dataset
        item['_scale'] = scale
        item['_trans'] = trans
        item['_trans_lh'] = trans_lh
        item['_trans_rh'] = trans_rh

        return item

    def __len__(self):
        return self.length
        
       
