import os
import argparse
from PIL import Image, ImageDraw


def txtRead(file_name):
    with open(file_name, 'r') as f:
        outList = f.readlines()
    outList = [a.strip() for a in outList]
    return outList


def draw_bbox(draw_img, anno_list, rgb=(255, 0, 0)):
    bbox_cls = anno_list[0]
    bbox = [float(a) for a in anno_list[4:8]]

    draw_img.rectangle(bbox, outline=rgb, width=1)
    draw_img.text(bbox[:2], bbox_cls)


def parse_args():
    parser = argparse.ArgumentParser("KITTI draw bbox script")
    parser.add_argument(
        '--result_dir',
        type=str,
        default='./output_results',
        help='detection result directory to evaluate')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./kitti3d',
        help='KITTI dataset root directory')
    parser.add_argument('--out_dir', type=str, default='./display', help='')
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        help='evaluation split, default val')
    parser.add_argument(
        '--class_name',
        type=str,
        default='Car',
        help='evaluation class name, default Car')
    args = parser.parse_args()
    return args


def kitti_draw_img():
    args = parse_args()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    dataDir = args.data_dir
    resultDir = args.result_dir
    imgDir = os.path.join(dataDir, "training/image_2")
    labelDir = os.path.join(dataDir, "training/label_2")
    txtFile = os.path.join(dataDir, f"ImageSets/{args.split}.txt")
    txtList = txtRead(txtFile)

    outDir = args.out_dir
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    for i, name in enumerate(txtList):
        print(i, name)
        img = Image.open(os.path.join(imgDir, name + '.png'))
        draw = ImageDraw.Draw(img)

        gt_annos = txtRead(os.path.join(labelDir, name + '.txt'))
        res_annos = txtRead(os.path.join(resultDir, name + '.txt'))

        # draw gt
        for gt_anno in gt_annos:
            draw_bbox(draw, gt_anno.split(' '), colors[0])

        # draw pred
        for res_anno in res_annos:
            draw_bbox(draw, res_anno.split(' '), colors[1])

        img.save(os.path.join(outDir, name + '.png'))


if __name__ == "__main__":
    kitti_draw_img()
    print('Done')
