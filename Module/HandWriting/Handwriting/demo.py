import json

import cv2
import matplotlib.pyplot as plt
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')
from path import Path

from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

def run(config, classes,image,output):
    result = ""

    with open(config) as f:
        sample_config = json.load(f)

    with open(classes) as f:
        word_list = [w.strip().upper() for w in f.readlines()]
    prefix_tree = PrefixTree(word_list)

    for decoder in ['best_path', 'word_beam_search']:
        img_filename = Path(image) 
        # print(f'Reading file {img_filename} with decoder {decoder}')

        # read text
        img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
        scale = sample_config[img_filename.basename()]['scale'] if img_filename.basename() in sample_config else 1
        margin = sample_config[img_filename.basename()]['margin'] if img_filename.basename() in sample_config else 0
        read_lines = read_page(img,
                                detector_config=DetectorConfig(scale=scale, margin=margin),
                                line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                                reader_config=ReaderConfig(decoder=decoder, prefix_tree=prefix_tree))

        # output text
        for read_line in read_lines:
            result += ' '.join(read_word.text for read_word in read_line)
            # print(' '.join(read_word.text for read_word in read_line))
            result += "\n"
        
        result += "\n"
        if (decoder=="best_path"):
            result += "-----Split-----"
            result += "\n\n"

        # plot image with detections and texts as overlay
        plt.figure(f'Image: {img_filename} Decoder: {decoder}')
        plt.imshow(img, cmap='gray')
        for i, read_line in enumerate(read_lines):
            for read_word in read_line:
                aabb = read_word.aabb
                xs = [aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin]
                ys = [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin]
                plt.plot(xs, ys, c='r' if i % 2 else 'b')
                plt.text(aabb.xmin, aabb.ymin - 2, read_word.text)
        plt.savefig(output+decoder+"_output.png",dpi=300,bbox_inches='tight')
    # plt.show()
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./data/config.json')
    parser.add_argument('--classes',type=str,default="./data/words_alpha.txt")
    parser.add_argument('--image',type=str,default="")
    parser.add_argument('--output',type=str,default="")
    args = parser.parse_args()
    result = run(args.config,args.classes,args.image,args.output)
    print(result)

