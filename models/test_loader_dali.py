from nvidia.dali import pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch

@pipeline_def
def my_pipeline(file_root='/root/autodl-tmp/B1_fudan_univ/slide_roi_for_B1_C1'):
    img_files, _ = fn.readers.file(file_root=file_root)
    # to gpu, to rgb
    imgs = fn.decoders.image(img_files, output_type=types.RGB, device='mixed')
    # resize
    imgs = fn.resize(imgs, resize_longer=608)
    # / 255
    imgs = imgs / 255
    # to chw
    imgs = fn.transpose(imgs, perm=[2,0,1])
    return imgs

def get_dali_iter(file_root='/root/autodl-tmp/B1_fudan_univ/slide_roi_for_B1_C1', batch_size=4, num_threads=2, device_id=0):
    pipe = my_pipeline(file_root, batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    pipe.build()
    dali_iter = DALIGenericIterator(pipe, ['img'], last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True)
    return dali_iter

if __name__ == "__main__":
    loader = get_dali_iter(batch_size=16)
    for i, data in enumerate(loader):
        # input = data[0]['img']
        # print(input.shape)
        print(i*16)
# 第三章 蒸馏、剪枝、缩小方法
# 三个到五个任务
# 公开数据集 单个任务 三个以上的任务，拷贝结果
# 第四章最后一节 公开数据集 对比单任务的能力
# 毕业论文，所有数据集
# 大模型与多任务