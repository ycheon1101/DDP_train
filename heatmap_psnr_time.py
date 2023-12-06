from pathlib import Path
import sys
# path
heatmap_path = Path(__file__).parent
src_path = heatmap_path.parent
# print(src_path)
sys.path.append(str(src_path) + '/modules')

import table_images
from mlp import MLP
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from positional_encoding import GaussianFourier
from PIL import Image
import seaborn as sns
import pandas as pd
import os
import time
from set_device import device


def main():

    

    # params 
    learning_rate = 5e-3
    num_epochs = 500
    max_pixel = 1.0

    # create layer and neuron list
    neuron_list = [16, 32, 64, 128]
    hidden_layer_list = [1, 2, 4, 6]
    # neuron_list = [256, 512]
    # hidden_layer_list = [8, 10]

    # get images
    img_df, crop_size = table_images.make_table()

    # get a image
    for image in range(1, len(img_df.index) + 1):
        
        # create a new directory to store image
        dir_path = Path(src_path) / 'images' / 'tested_images' / 'heatmap_test'
        img_dir = f'output_image_{image}'
        os.makedirs(dir_path / img_dir, exist_ok=True)


        # set target
        target = img_df['img_flatten'][image].to(device)

        # set xy coord
        xy_flatten = img_df['xy_flatten'][image].to(device)

        # positional encoding
        fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 64, scale=4)(xy_flatten).to(device)
        
        # calc loss
        criterion = nn.MSELoss()

        # initialize psnr_list and train_time_list
        psnr_list = np.zeros((len(neuron_list), len(hidden_layer_list)))
        train_time_list = np.zeros((len(neuron_list), len(hidden_layer_list)))

        # train from num_neuron * num_hidden_layer
        for layer in range(len(hidden_layer_list)):
            # torch.cuda.empty_cache()
            for neuron in range(len(neuron_list)):

                # start time
                start_time = time.time()

                model = MLP(in_feature=128, hidden_feature=neuron_list[neuron], hidden_layers=hidden_layer_list[layer], out_feature=3).to(device)

                optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

                for epoch in range(num_epochs):

                    generated = model(fourier_result)
                    loss = criterion(generated, target)

                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(f"Image{image}, Epoch {epoch+1}/{num_epochs}, layer: {hidden_layer_list[layer]}, neuron: {neuron_list[neuron]}, Loss: {loss.item()}")

                    
                
                # end time
                end_time = time.time()

                # calc train time
                train_time = end_time - start_time

                print(f'Image{image} trian time: {train_time}')

                train_time_list[layer, neuron] = train_time

                # calc psnr
                calc_psnr = 10 * torch.log10(max_pixel ** 2 / loss)
                psnr_list[layer, neuron] = calc_psnr

                # Getting the output from the network and reshaping it
                generated_img = model(fourier_result)
                generated_reshape = torch.reshape(generated_img, (crop_size, crop_size, 3))

                generated_reshape = generated_reshape * 255.0
                generated_reshape = generated_reshape.cpu().detach().numpy()

                # save image
                # img_path = os.path.join(dir_path / img_dir, f'reconstructed_image_{hidden_layer_list[layer]}_{neuron_list[neuron]}.jpg')
                # save_img = Image.fromarray(generated_reshape.astype(np.uint8))
                # save_img.save(img_path)

        # make table
        hidden_layer_df = pd.DataFrame(psnr_list, index=hidden_layer_list, columns=neuron_list)
        train_time_df = pd.DataFrame(train_time_list, index=hidden_layer_list, columns=neuron_list)

        # generate psnr heatmap and save
        plt.figure(figsize=(15, 8))
        heat_map_psnr = sns.heatmap(hidden_layer_df, cmap='GnBu', annot=False, annot_kws={'size':10}, fmt='.3f', xticklabels=True, yticklabels=True)
        plt.xlabel('neurons')
        plt.ylabel('layers')
        plt.title('psnr_heatmap_img' + str(image))

        # heatmap_image_path = os.path.join(dir_path / img_dir, f'psnr_heatmap_image_{image}.jpg')
        # heat_map_psnr.figure.savefig(heatmap_image_path)
        plt.close()

        # generate time heatmap and save
        plt.figure(figsize=(15, 8))
        heat_map_time = sns.heatmap(train_time_df, cmap='GnBu', annot=True, annot_kws={'size':10}, fmt='.3f', xticklabels=True, yticklabels=True, vmin=0, vmax=50)
        plt.xlabel('neurons')
        plt.ylabel('layers')
        plt.title('time_heatmap_single_gpu')

        heatmap_image_path = os.path.join(dir_path / img_dir, f'time_heatmap_image_{image}_single_ex2.jpg')
        heat_map_time.figure.savefig(heatmap_image_path)



if __name__ == '__main__':
    main()




   