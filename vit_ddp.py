import os
import torch
from torch import nn
from torchvision.transforms import ToTensor
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
import datetime
from torch.profiler import profile, record_function


# device = "cuda:1"
torch.manual_seed(0)

import dataset

# param 
img_size = 28 * 28
patch_size = 4
in_channel = 1
atten_head = 8
num_classes = 10
num_layers = 5
emb_dim = 16
learning_rate = 0.001

# global_data_sent = 0
global_forward_data_processed = 0
global_backward_data_processed = 0

# def ddp_comm_hook(state, bucket):
#     global global_data_sent
#     for tensor in bucket.get_per_parameter_tensors():
#         tensor_size = tensor.numel() * tensor.element_size()
#         global_data_sent += tensor_size
#     return dist.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook(state, bucket)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_ADDR'] = 'fe80::e017:49ff:fe15:d163'
    # os.environ['MASTER_ADDR'] = 'fe80::e017:49ff:fe15:d163'
    # os.environ['MASTER_ADDR'] = 'tcp://172.30.0.7'
    # os.environ['MASTER_ADDR'] = '172.30.0.7'
    # os.environ['MASTER_ADDR'] = '192.168.0.32'
    os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

class EmbeddingLayer(nn.Module):
    def __init__(self, in_chan, img_size, patch_size):
        super().__init__()
        self.num_patches = int(img_size / pow(patch_size, 2)) # 49
        self.emb_size = in_chan * patch_size * patch_size # 16
        self.project = nn.Conv2d(in_chan, self.emb_size, kernel_size= patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,self.emb_size))
        self.positions = nn.Parameter(torch.randn(self.num_patches+ 1, self.emb_size)) # [50,16]
        # self.positions = self.create_sinusoidal_embeddings(self.num_patches + 1, self.emb_size).to(device)
    
    def create_sinusoidal_embeddings(self, n_patches, dim):
        position = np.arange(n_patches)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        sinusoidal_emb = np.zeros((n_patches, dim))
        sinusoidal_emb[:, 0::2] = np.sin(position * div_term)
        sinusoidal_emb[:, 1::2] = np.cos(position * div_term)
        return torch.tensor(sinusoidal_emb, dtype=torch.float32)
    
 
    def forward(self, x):
        x = self.project(x)
        x = x.view(-1, 49, 16) # [batch_size, 49, 16]
        repeat_cls = self.cls_token.repeat(x.size()[0],1,1) #[batch_size, 1 , 16]
        x = torch.cat((repeat_cls, x), dim=1)
        x += self.positions
        return x

class Multihead(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.multiheadattention = nn.MultiheadAttention(emb_size, num_heads, batch_first = True, dropout=0.2)
        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attn_output, attention = self.multiheadattention(query, key, value)
        return attn_output, attention

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, hidden = 4, drop_p = 0.2):
        super().__init__(
            nn.Linear(emb_size, hidden * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden * emb_size, emb_size)
        )

class VIT(nn.Module):
    def __init__(self,emb_size = emb_dim):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(in_channel, img_size, patch_size)
        self.Multihead = Multihead(emb_size, atten_head)
        self.feed_forward = FeedForwardBlock(emb_size)
        self.norm = nn.LayerNorm(emb_size)
        
    def forward(self, x):
        global global_forward_data_processed

        # embedding
        x = self.embedding_layer(x)

        # normalize
        norm_x = self.norm(x)

        # multihead
        multihead_output, attention = self.Multihead(norm_x)
        
        # residual
        output = multihead_output + x

        # normalize
        norm_output = self.norm(output)

        # mlp(FeedForward)
        feed_forward = self.feed_forward(norm_output)
        
        # residual
        final_out = feed_forward + output

        if self.training:
            tensor_size = x.numel() * x.element_size()
            global_forward_data_processed += tensor_size
        
        return final_out, attention

# the number of encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers = num_layers, num_classes = num_classes):
        super().__init__()
        self.layers = nn.ModuleList([VIT() for _ in range(num_layers)])
        self.classifier = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            final_out, _ = layer(x)
        final_out = final_out.mean(dim=1)
        final_out = self.classifier(final_out)
            
        return final_out

def calculate_data_transferred(tensor):
    return tensor.nelement() * tensor.element_size()

def log_data_transfer(tensor, action):
    data_size = tensor.nelement() * tensor.element_size()
    print(f"{action}, Data Size: {data_size} bytes")


def train(rank, world_size):
    global global_forward_data_processed, global_backward_data_processed

    # setup
    setup(rank, world_size)

    if rank == 0:
        # device = torch.device("cuda:1")
        device = torch.device("cuda:0")
    elif rank == 1:
        # device = torch.device("cuda:2")
        device = torch.device("cuda:1")


    model = TransformerEncoder().to(device)
    # model = DDP(model, device_ids=[rank + 1], find_unused_parameters=True)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    for param in model.parameters():
        dist.broadcast(param.data, src=0) 

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 30

    # model.register_comm_hook(state=None, hook=ddp_comm_hook)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()

        global_forward_data_processed = 0
        global_backward_data_processed = 0

        forward_data_transferred = 0
        backward_data_transferred = 0

        total_gradient_size = 0

        # with profile(activities=[torch.profiler.ProfilerActivity.CPU, 
        #                           torch.profiler.ProfilerActivity.CUDA], 
        #              record_shapes=True) as prof:
        # global_data_sent = 0
        # epoch_start = datetime.datetime.now()
        # print(f'Epoch {epoch+1} start at {epoch_start}')
        for images, labels in dataset.mnist_train_dataloader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            start_forward = time.time()
            outputs = model(images)
            end_forward = time.time()

            forward_data_transferred += calculate_data_transferred(images)

            start_backward = time.time()
            loss = criterion(outputs, labels)
            # with record_function("model_forward"):
            #         outputs = model(images)
            #         loss = criterion(outputs, labels)

            # Backward and optimize
            
            optimizer.zero_grad()

            if model.training:
                tensor_size = loss.numel() * loss.element_size()
                global_backward_data_processed += tensor_size


            loss.backward()
            # for param in model.parameters():
            #     if param.requires_grad and param.grad is not None:
            #         log_data_transfer(param.grad, "All-reduce")

            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_size = param.grad.nelement() * param.grad.element_size()
                    total_gradient_size += grad_size

            end_backward = time.time()

            backward_data_transferred += calculate_data_transferred(loss)

            optimizer.step()

            # with record_function("model_backward"):
            #         optimizer.zero_grad()
            #         loss.backward()
            #         if model.training:
            #             tensor_size = loss.numel() * loss.element_size()
            #             global_backward_data_processed += tensor_size
            #         optimizer.step()
    
        # epoch_end = datetime.datetime.now()
        # print(f'Epoch {epoch+1} end at {epoch_end}')

        print(f"Epoch {epoch+1}, Forward Time: {end_forward - start_forward}, Data Transferred: {forward_data_transferred} bytes")
        print(f"Epoch {epoch+1}, Backward Time: {end_backward - start_backward}, Data Transferred: {backward_data_transferred} bytes")
        
        # print(f'Epoch {epoch+1}: Forward data processed = {global_forward_data_processed} bytes, Backward data processed = {global_backward_data_processed} bytes')
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        print(f"Epoch {epoch+1}: Total Gradient Size for All-reduce: {total_gradient_size} bytes")
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        # print(f"Epoch {epoch+1} finished, total data sent: {global_data_sent} bytes")
    
    end_time = time.time()

    train_time = end_time - start_time
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, labels in dataset.mnist_test_dataloader:
                    image, labels = image.to(device), labels.to(device)
                    
                    outputs = model(image)
                    predicted = outputs.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy}%')
    print(train_time)
    cleanup()


def main():
    world_size = 2
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=0)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    cleanup()


if __name__ == '__main__':
    main()

