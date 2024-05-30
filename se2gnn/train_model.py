from models.model import SO2MessagePassing, InvariantMessagePassing


import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
import gc
import pickle
import torch
from torch_geometric.data import InMemoryDataset

from torch_geometric.data.collate import collate
from torch_geometric.seed import seed_everything
from models.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from torch_geometric.utils import scatter

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique 
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" 
    init_process_group(backend="nccl", rank = rank, world_size=world_size)
    
    
def get_random_training_test_data(u,v, batch , time_slice = 2):
    start_ind = torch.multinomial(torch.ones(14-time_slice), batch.max().item() + 1, replacement=True).to(u.device)
    start_ind = start_ind[batch]
    ind_train = torch.arange(0,time_slice, device = u.device).reshape(1,-1) + start_ind.reshape(-1,1)
    ind_test =  start_ind + time_slice
    u_train = u.gather(1,ind_train)
    u_test = u.gather(1,ind_test.unsqueeze(1))
    ind_v_train = torch.stack((ind_train,ind_train),dim=1)
    ind_v_test = torch.stack((ind_test,ind_test),dim=1)
    v_trans = v.transpose(1,2)
    v_train = v_trans.gather(2,ind_v_train).transpose(1,2)
    v_test = v_trans.gather(2,ind_v_test.unsqueeze(-1)).transpose(1,2)
    return u_train, u_test, v_train,v_test.squeeze(1)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,        
        val_data: DataLoader, 
        test_data: DataLoader,      
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        time_slice = 2,
        name = ""
    ) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = 10
        self.test_every = 1
        self.time_slice = 3
        self.name = name
        self.scheduler = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer, warmup_epochs=20,max_epochs=500, warmup_start_lr=1e-8, eta_min=3e-6)

    def _run_batch(self, data: Data):
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()  # Clear gradients.   
        
        u,u_t, v, v_t = get_random_training_test_data(data.u_train, data.v_train, data.batch, time_slice = self.time_slice)

        
        u_out, v_out = self.model(u = u,
                            v = v, 
                            boundary_norm = data.norm, 
                            is_boundary = data.is_boundary, 
                            y_force = data.force, 
                            pos = data.pos, 
                            edge_index = data.edge_index, batch = data.batch)
        u_t = u_t.flatten()
        a = v_out -v_t
   
        loss_v = ((v_out -v_t)**2).sum(dim = 1)
        loss_u = (u_out-u_t)**2

        
        loss = (loss_v + loss_u).mean()
        
        loss.backward()
        self.optimizer.step()  # Update parameters based on gradients.
        return loss.item()

    def _run_epoch(self, epoch: int):
        b_sz = next(iter(self.train_data))#.num_graphs
        b_sz = b_sz.num_graphs
        
        loss_tots = []

        for i, data in enumerate(self.train_data):                   
            data = data.to(self.gpu_id)
            loss_tot = self._run_batch(data)
            loss_tots.append(loss_tot)

        return sum(loss_tots) / len(loss_tots)
        
        
    def _save_best_model(self):
        ckp = self.model.module.state_dict()
        PATH = "saved_models/best_model_" + self.name + ".pkl"
        torch.save(ckp, PATH)
        print(f"best model saved")        

    def _load_best_model(self):

        PATH = "saved_models/best_model_" + self.name + ".pkl"
        
        self.model.module.load_state_dict(torch.load(PATH))

        print(f"best model loaded")  
        
    def train(self, max_epochs: int):
        val_min = 1.e6
        train_errors = []
        val_errors = []
        test_error = 0.
        for epoch in range(max_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            
            loss_tot = self._run_epoch(epoch)
            
            train_errors.append(loss_tot)
            if (self.gpu_id == 0):
                val_loss = self.test(self.val_data)
                wandb.log({"step": epoch, "val_loss": val_loss, "train_loss": loss_tot})
                val_errors.append(val_loss)
                print("Epoch: " +str(epoch)+ " Val loss " +  str(val_loss) )
                if val_loss < val_min:
                    val_min = val_loss
                    self._save_best_model()
            
                print("Epoch: " +str(epoch)+ " Train Loss tot: " + str(loss_tot))
            self.scheduler.step()
        if (self.gpu_id == 0):
            self._load_best_model()
            test_error = self.test(self.test_data)
            print("Test loss " +  str(test_error) )
        
        return test_error, val_errors, train_errors
        
         
 
                
    def test(self, data):
        losses = []
        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            correct = 0
            for data in data:
                for i in range(14-self.time_slice):
                    data = data.to(self.gpu_id)
                    u = data.u_train[:,i:i+self.time_slice]
                    v = data.v_train[:,i:i+self.time_slice,:]
                    u_t = data.u_train[:,i+self.time_slice]
                    v_t = data.v_train[:,i+self.time_slice,:]
                     
                    
            
                    u_out, v_out = self.model(u = u,
                                        v = v, 
                                        boundary_norm = data.norm, 
                                        is_boundary = data.is_boundary, 
                                        y_force =data.force, 
                                        pos = data.pos, 
                                        edge_index = data.edge_index, batch = data.batch)
                                        
                    loss_v = ((v_out - v_t)**2).sum(dim = 1)
                    loss_u = (u_out-u_t.flatten())**2
                    loss = (loss_v + loss_u).mean()
                    losses.append(loss.item())
            test_loss = sum(losses) / len(losses )
            return test_loss
      
def get_data(link_to_train_data, link_to_testdata):
    seed_everything(0)
    train_ = InMemoryDataset()
    data, slices = torch.load(link_to_train_data)
    train_.data = data
    train_.slices = slices
    
    train_ = train_.shuffle()
    index = len(train_)-round(len(train_)*0.05)
    train = train_[:index]
    val = train_[index:]
        
    test = InMemoryDataset()
    data, slices = torch.load(link_to_testdata)
    test.data = data
    test.slices = slices    


    return train,val,test
    
def get_model(is_se2 = True, L_max = 12, num_rep = 9,num_layers = 3, n_scalars = 32 ):

    if is_se2:
      model = SO2MessagePassing(time_slize = 3, L_max = L_max, num_rep = num_rep, num_layers = num_layers, n_scalars = n_scalars, rot_feed_forward = True, updated_node_embedding=True)
    else:
      model = InvariantMessagePassing(time_slize = 3, num_layers = num_layers, n_scalars = n_scalars)
    return model
    
def prepare_dataloader(dataset: Dataset, batch_size: int, world_size: int, rank: int):
    train_sampler = DistributedSampler(dataset, num_replicas=world_size,
                                       rank=rank)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler = train_sampler
    )

def prepare_test_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False)

def main(rank: int, world_size: int, batch_size: int, is_se2: bool, L_max: int, num_rep: int, num_layers: int, n_scalars: int, link_to_train_data: str, link_to_testdata: str):
    ddp_setup(rank, world_size)
    total_epochs = 500
    
    train, val, test = get_data(link_to_train_data, link_to_testdata)
    train_data = prepare_dataloader(train, batch_size, world_size, rank)
    test_data= prepare_test_dataloader(test, 32)   
    val_data= prepare_test_dataloader(val, 32)     
    
      
    name = "model_name"
    
    model = get_model(is_se2 = is_se2, L_max = L_max, num_rep = num_rep,num_layers = num_layers, n_scalars = n_scalars)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
    trainer = Trainer(model, train_data, val_data, test_data,  optimizer, gpu_id = rank, name = name)
    test_error, val_errors, train_errors = trainer.train(total_epochs)


    destroy_process_group()


if __name__ == "__main__":
    import argparse
    print("Cuda is available: " + str(torch.cuda.nccl.is_available(torch.randn(1).cuda())))
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=1, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 8)')
    parser.add_argument('--num_layers', default=7, type=int, help='Number of layers (default: 7)')
    parser.add_argument('--num_scalars', default=64, type=int, help='Number of layers (default: 64)')
    parser.add_argument('--num_rot_rep', default=64, type=int, help='Number rotation representations (default: 64)')
    parser.add_argument('--test_data', default="", type=str, help='Link to test data')
    parser.add_argument('--train_data', default="", type=str, help='Link to train data')
    parser.add_argument("--SE2", action="store_false", help="use SE2 model")
    
    args = parser.parse_args()
    world_size = torch.cuda.device_count()

        
    mp.spawn(main, args = (world_size, args.batch_size, args.SE2, L_max, args.num_rot_rep, args.num_layers, args.num_scalars, args.train_data, args.test_data ), nprocs = world_size)
            
