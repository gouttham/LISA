import glob
import json
import os
import random
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST

def init_xbd_pre(base_image_dir):
    with open("utils/cd_classes/xbd_classes_pre.json", "r") as f:
        xbd_classes = json.load(f)
    xbd_classes = np.array(xbd_classes)

    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "xbd", "images"))
    )
    xbd_image_ids = []
    for x in image_ids:
        if x.endswith("pre_disaster.png"):
            xbd_image_ids.append(x[:-4])

    xbd_images = []
    for image_id in xbd_image_ids:  # self.descriptions:
        xbd_images.append(
            os.path.join(
                base_image_dir, "xbd", "images", "{}.png".format(image_id),
            )
        )
    xbd_labels = [
        x.replace("images", "targets").replace(".png", "_target.png")
        for x in xbd_images
    ]
    print("xbd images = {}, xbd targets = {}, xbd pre disaster classes = {}".format(len(xbd_images), len(xbd_labels), xbd_classes))
    print()
    return xbd_classes, xbd_images, xbd_labels


def init_xbd_post(base_image_dir):
    with open("utils/cd_classes/xbd_classes_post.json", "r") as f:
        xbd_classes = json.load(f)
    xbd_classes = np.array(xbd_classes)

    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "xbd", "images"))
    )
    xbd_image_ids = []
    for x in image_ids:
        if x.endswith("post_disaster.png"):
            xbd_image_ids.append(x[:-4])

    xbd_images = []
    for image_id in xbd_image_ids:  # self.descriptions:
        xbd_images.append(
            os.path.join(
                base_image_dir, "xbd", "images", "{}.png".format(image_id),
            )
        )
    xbd_labels = [
        x.replace("images", "targets").replace(".png", "_target.png")
        for x in xbd_images
    ]
    print("xbd images = {}, xbd targets = {}, xbd post disaster classes = {}".format(len(xbd_images), len(xbd_labels), xbd_classes))
    print()
    return xbd_classes, xbd_images, xbd_labels


class CD_Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 5,
        exclude_val=False,
        sem_seg_data="xbd||ida-bd||whu-cd||levir-cd",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data
        self.sem_seg_datas = self.sem_seg_datas.replace("xbd", "xbd_pre||xbd_post").split("||")
        print(self.sem_seg_datas)
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        image, labels = self.data2list[ds]
        idx = random.randint(0, len(image) - 1)
        image_path = image[idx]
        label_path = labels[idx]
        # print("image path - ", image_path)
        # print("label path - ", label_path)
        label = Image.open(label_path)
        label = np.array(label)

        img_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        unique_label = np.unique(label).tolist()
        if 255 in unique_label:
            unique_label.remove(255)
        if len(unique_label) == 0:
            return self.__getitem__(0)

        # print(unique_label)

        classes = [self.data2classes[ds][class_id] for class_id in unique_label]
        # print(classes)
        if len(classes) >= self.num_classes_per_sample:
            sampled_classes = np.random.choice(
                classes, size=self.num_classes_per_sample, replace=False
            ).tolist()
        else:
            sampled_classes = classes

        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))

            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids.append(class_id)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        # conv = conversation_lib.get_default_conv_template("vicuna")


        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        label = torch.from_numpy(label).long()
        masks = []
        for class_id in class_ids:
            masks.append(label == class_id)
        masks = torch.stack(masks, dim=0)

        # print(conversations)
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )


def init_cd_dset_xbd(base_image_dir):
    #   REQUIRED: directory structure 
    #
    #   base dir
    #   |
    #   |_____ xbd
    #           |____ images
    #           |____ targets
    #                   
    with open("utils/cd_classes/xbd_classes_post.json", "r") as f:
        xbd_classes = json.load(f)
    xbd_classes = np.array(xbd_classes)

    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "xbd", "images"))
    )
    xbd_image_ids_pre = []
    for x in image_ids:
        if x.endswith("pre_disaster.png"):
            xbd_image_ids_pre.append(x[:-4])

    xbd_images_pre = []
    for image_id in xbd_image_ids_pre:  # self.descriptions:
        xbd_images_pre.append(
            os.path.join(
                base_image_dir, "xbd", "images", "{}.png".format(image_id),
            )
        )

    xbd_images_post = [
        x.replace("_pre_", "_post_")
        for x in xbd_images_pre
    ]

    for i in range(len(xbd_images_post)):
        assert(os.path.isfile(xbd_images_pre[i]))
        assert(os.path.isfile(xbd_images_post[i]))
        assert(xbd_images_post[i].replace("_post_", "_pre_") == xbd_images_pre[i])

    xbd_labels = [
        x.replace("images", "targets").replace(".png", "_target.png")
        for x in xbd_images_post
    ]
    print("xbd images = {}, xbd targets = {}, xbd post disaster classes = {}".format(len(xbd_images_pre), len(xbd_labels), xbd_classes))
    print()
    return xbd_classes, xbd_images_pre, xbd_images_post, xbd_labels


class Contrastive_CD_Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 5,
        exclude_val=False,
        sem_seg_data="xbd||ida-bd||whu-cd||levir-cd",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        # self.sem_seg_datas = self.sem_seg_datas.replace("xbd", "xbd_pre||xbd_post").split("||")
        print(self.sem_seg_datas)
        for ds in self.sem_seg_datas:
            classes, pre_images, post_images, labels = eval("init_cd_dset_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (pre_images, post_images, labels)
            self.data2classes[ds] = classes

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


    def load_pre_post_img(self, paths, idx):
        image_path = paths[idx]
        print(image_path)

        img_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        return image_path, image, image_clip, resize
        
        
    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        pre_images, post_images, labels = self.data2list[ds]
        assert(len(pre_images) == len(post_images))

        idx = random.randint(0, len(pre_images) - 1)

        # loading in the pre and post images
        pre_image_path, pre_image, pre_image_clip, pre_resize = self.load_pre_post_img(pre_images, idx)
        post_image_path, post_image, post_image_clip, post_resize = self.load_pre_post_img(post_images, idx)

        # Loading in the labels
        label_path = labels[idx]
        label = Image.open(label_path)
        label = np.array(label)
        unique_label = np.unique(label).tolist()
        if 255 in unique_label:
            unique_label.remove(255)
        if len(unique_label) == 0:
            return self.__getitem__(0)

        classes = [self.data2classes[ds][class_id] for class_id in unique_label]
        if len(classes) >= self.num_classes_per_sample:
            sampled_classes = np.random.choice(
                classes, size=self.num_classes_per_sample, replace=False
            ).tolist()
        else:
            sampled_classes = classes

        questions = []
        answers = []
        class_ids = []
        print(sampled_classes)

        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))

            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids.append(class_id)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        # conv = conversation_lib.get_default_conv_template("vicuna")

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        label = torch.from_numpy(label).long()
        masks = []
        for class_id in class_ids:
            masks.append(label == class_id)
        masks = torch.stack(masks, dim=0)

        print(conversations)
        return (
            (pre_image_path, post_image_path),
            (pre_image, post_image),
            (pre_image_clip, post_image_clip),
            conversations,
            masks,
            label,
            (pre_resize, post_resize),
            questions,
            sampled_classes,
        )


if __name__ == '__main__':
    xbd = Contrastive_CD_Dataset(sys.argv[1], None, None, sem_seg_data="xbd")
    data = xbd[0]