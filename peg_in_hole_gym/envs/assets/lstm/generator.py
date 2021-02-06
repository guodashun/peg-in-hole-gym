import numpy as np
import os
import random
from .flyer import Flyer
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d



class Generator():
    data_form_list = ['Banana']
    def __init__(self, data_form, pre_num=30):
        assert data_form in self.data_form_list, f'Cannot generate {data_form} data!'
        self.data_form = data_form
        self.flyer = Flyer(data_form)
        self.samples = None
        self.pre_num = pre_num
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.sim_model_dir = f'{cur_dir}/'
        self.real_model_dir = f'{cur_dir}/'

        data_dir = f'{cur_dir}/data/{self.data_form}/train'
        data_list = os.listdir(data_dir)
        self.samples = [self.flyer.load_data(f'{data_dir}/{i}')[:self.pre_num] for i in data_list]

    def generate(self, sigma=0.002):
        # random choice
        begin = self.samples[random.randint(0, len(self.samples) - 1)]
        # self.plt_show(begin, color=['green'])

        # add noise
        begin = np.array([[j + random.gauss(0, sigma) for j in i] for i in begin])
        # self.plt_show(begin)

        self.flyer.set_model(self.sim_model_dir)
        




    @classmethod
    def plt_show(cls, data, new_data, color=['red']):
        ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
        ax.set_title('3d_image_show')  # 设置本图名称
        ax.set_xlabel('X')  # 设置x坐标轴
        ax.set_ylabel('Y')  # 设置y坐标轴
        ax.set_zlabel('Z')  # 设置z坐标轴
        for i in range(len(data)):
            ax.scatter(data[i][0], data[i][1], data[i][2], color=color[0], s=1)

        for i in range(len(new_data)):
            ax.scatter(new_data[i][0], new_data[i][1], new_data[i][2], color=color[0], s=1)
            # ax.plot(data[i][0], data[i][1], data[i][2], color=color[i])
        plt.show()


if __name__ == '__main__':
    gen = Generator("Banana")
    gen.generate()
