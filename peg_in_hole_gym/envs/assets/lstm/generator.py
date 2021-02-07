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
        self.pre_num = pre_num
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.sim_model_dir = f'{cur_dir}/output/{self.data_form}_sim_model.pkl'
        self.real_model_dir = f'{cur_dir}/output/{self.data_form}_real_model.pkl'

        self.data_dir = f'{cur_dir}/data/{self.data_form}/train'
        self.data_list = os.listdir(self.data_dir)

    def generate(self, sigma=0.02):
        # random choice
        begin = self.flyer.load_data(f'{self.data_dir}/{self.data_list[random.randint(0, len(self.data_list) - 1)]}')[:self.pre_num]

        # add noise
        begin_noised = np.array([[j + random.gauss(0, sigma) for j in i] for i in begin])

        self.flyer.set_model(self.real_model_dir)
        self.flyer.init_lstm_pre(begin_noised)
        for _ in range(80):
            self.flyer.get_next_pre_data()
        data1 = self.flyer.get_whole_pre_data()

        self.flyer.set_model(self.sim_model_dir)
        self.flyer.init_lstm_pre(begin_noised)
        for _ in range(80):
            self.flyer.get_next_pre_data()
        data2 = self.flyer.get_whole_pre_data()

        err = np.linalg.norm(data1[-1, :3] - data2[-1, :3])
        # self.plt_show(begin, begin_noised)
        # self.plt_show(data1, data2)
        return err



    @classmethod
    def plt_show(cls, data, new_data, color=['red', 'green']):
        ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
        ax.set_title('3d_image_show')  # 设置本图名称
        ax.set_xlabel('X')  # 设置x坐标轴
        ax.set_ylabel('Y')  # 设置y坐标轴
        ax.set_zlabel('Z')  # 设置z坐标轴
        for i in range(len(data)):
            ax.scatter(data[i][0], data[i][1], data[i][2], color=color[0], s=1)

        for i in range(len(new_data)):
            ax.scatter(new_data[i][0], new_data[i][1], new_data[i][2], color=color[1], s=1)
            # ax.plot(data[i][0], data[i][1], data[i][2], color=color[i])
        plt.show()


if __name__ == '__main__':
    gen = Generator("Banana")
    sum_err = 0.
    for i in range(200):
        err = gen.generate(0.06)
        print(f"traj {i} err is {err}")
        sum_err += err
    print(f"avg err is {sum_err/200}")
