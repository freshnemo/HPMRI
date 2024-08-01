import time
import pandas as pd 
import numpy as np
import os 
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from modules import NMR_matrix, NMR_mask_matrix, return_NMR_pad, return_pad, check_max_len_img, FindCorner, FindNonzero2D
from modules import return_img_pad, check_max_len, check_max_len_img_hp, return_ph_pad, LoadData, mult_attn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torch.nn import MultiheadAttention,Linear, ReLU, RNN,Sigmoid,Softmax
import torch 
class CCBTP_data(Dataset):

    def __init__(self, mouse_index, HPMRI, NMR, NMR_mask,sag_path,cor_path,ax_path,hp_path ):
        
        self.mouse_index = mouse_index
        self.HPMRI=HPMRI
        self.NMR=NMR
        self.NMR_mask=NMR_mask
        self.sag_path=sag_path

        self.cor_path=cor_path
        self.ax_path=ax_path
        self.sag_max, self.sag_loc =check_max_len_img(sag_path)
        self.cor_max, self.cor_loc=check_max_len_img(cor_path)
        self.ax_max, self.ax_loc=check_max_len_img(ax_path)
        self.max_seq=check_max_len(HPMRI)
        
        
        self.hp_path=hp_path
        self.hp_max_r, self.hp_max_c= check_max_len_img_hp(hp_path) #['agg_index']
        

    def __len__(self):
        return len(self.mouse_index)

    def __getitem__(self, idx):
        # Get locations of a fixed 2d area working for ax, sag and cor 
        loc_all = []
        for i in range(4):
            if i%2==0:
                loc_all.append(min(self.sag_loc[i],self.cor_loc[i],self.ax_loc[i]))
            else:
                loc_all.append(max(self.sag_loc[i],self.cor_loc[i],self.ax_loc[i]))
        mouse=self.mouse_index.iloc[idx].Mouse_Name

        coh=self.mouse_index.iloc[idx].Cohort

        mouse_class = self.mouse_index.iloc[idx].Label

        pt_HPMRI=self.HPMRI[self.HPMRI.Mouse_Name==mouse]

        pt_HPMRI=pt_HPMRI.sort_values(by='Days_Elapsed').drop(columns=['Mouse_Name','Cohort','Days_Elapsed']).values
        pt_HPMRI=return_pad(pt_HPMRI,self.max_seq)
        pt_NMR=self.NMR[self.NMR.Mouse_Name==mouse]
        pt_NMR=pt_NMR.sort_values(by='Days_Elapsed').drop(columns=['Mouse_Name','Cohort','Days_Elapsed']).values
        
        pt_NMR=return_pad(pt_NMR,self.max_seq)

        pt_NMR_mask=self.NMR_mask[self.NMR_mask.Mouse_Name==mouse]
        pt_NMR_mask=pt_NMR_mask.sort_values(by='Days_Elapsed').drop(columns=['Mouse_Name','Cohort','Days_Elapsed']).values
        pt_NMR_mask=return_pad(pt_NMR_mask,self.max_seq)
    
        pt_sag=self.sag_path[self.sag_path.Mouse_Name==mouse]
        pt_sag=return_img_pad (pt_sag,self.max_seq,self.sag_max,loc_all)
        pt_cor=self.cor_path[self.cor_path.Mouse_Name==mouse]
        pt_cor=return_img_pad (pt_cor,self.max_seq,self.cor_max,loc_all)
        pt_ax=self.ax_path[self.ax_path.Mouse_Name==mouse]
        pt_ax=return_img_pad(pt_ax,self.max_seq,self.ax_max,loc_all)  



        # ht_hp, raw data
        #print('pt_hp', self.hp_path.shape)      
        pt_hp=self.hp_path[self.hp_path.Mouse_Name==mouse]
        pt_hp=return_ph_pad(pt_hp, self.max_seq, self.hp_max_r, self.hp_max_c)
 
       
        return (pt_HPMRI,pt_NMR,pt_NMR_mask,pt_sag,pt_cor,pt_ax, mouse_class, pt_hp, mouse)
class CCBTP_attn(nn.Module):
    def __init__(self, config):
        super(CCBTP_attn, self).__init__()
        self.emb_size=config['emb_size']
        
        self.Lin_img_1=nn.Linear(config['img_pix_nu'],config['emb_size'],bias=False)
        self.Lin_img_2=nn.Linear(config['img_pix_nu'],config['emb_size'],bias=False)
        self.Lin_img_3=nn.Linear(config['img_pix_nu'],config['emb_size'],bias=False)
        
        self.Lin_img_all=nn.Linear(config['emb_size']*9,config['emb_size']*3)
        self.MuA_1=mult_attn(config).cuda()
        self.MuA_2=mult_attn(config).cuda()
        self.MuA_3=mult_attn(config).cuda()
        self.relu=ReLU()
        self.sigmoid=Sigmoid()
        self.Softmax=Softmax()



        self.conv1_3d = self._conv_layer_set_3d(1, 64)
        self.conv2_3d = self._conv_layer_set_3d(64, 128, h_size = 1)
        self.conv3_3d = self._conv_layer_set_3d(128, self.emb_size, h_size = 1)

        self.conv_hp1 = self._conv_layer_set_2d(1, 8)
        self.conv_hp2 = self._conv_layer_set_2d(8, 16)
        
        self.rnn_t=RNN(input_size=8,hidden_size=3,num_layers=config['rnn_layers'],
                    batch_first=True,dropout=config['drop_rate']
                    )
        self.rnn_hp=RNN(input_size=9,hidden_size=9,num_layers=config['rnn_layers'],
                    batch_first=True,dropout=config['drop_rate']
                    )
        
        self.rnn_hp_raw=RNN(input_size=280,hidden_size=config['rnn_hidden_size'],num_layers=config['rnn_layers'],
                    batch_first=True,dropout=config['drop_rate']
                    )
        
        self.final_lin2=nn.Linear(908,2,bias=True) 
        
        
    def forward(self,pt_HPMRI,pt_sag,pt_cor,pt_ax,pt_hp, NMR_data): 

        pt_sag=pt_sag.permute(1, 0,2,3,4)        
        pt_cor=pt_cor.permute(1, 0,2,3, 4)
        pt_ax=pt_ax.permute(1, 0,2,3,4)
        pt_hp=pt_hp.permute(1, 0,2,3)
        convs_size_sag, convs_size_cor, convs_size_ax= pt_sag.size(2), pt_cor.size(2), pt_ax.size(2)
        timestep=pt_ax.size()[0]
        batch=pt_ax.size()[1]
        input_rnn=torch.empty(size=(timestep*3, batch ,self.emb_size))
        input_rnn_hp=torch.empty(size=(timestep, batch ,280))
        for i in range(timestep):

            # 3d for images
            pt_sag_temp=pt_sag[i].float()
            pt_sag_temp = self.convs_3d(pt_sag_temp)

            pt_cor_temp=pt_cor[i].float()
            pt_cor_temp = self.convs_3d(pt_cor_temp)
            
            pt_ax_temp=pt_ax[i].float()
            pt_ax_temp = self.convs_3d(pt_ax_temp)


            pt_hp_temp=pt_hp[i].float().unsqueeze(1)
            pt_hp_temp = self.conv_hp1(pt_hp_temp)
            pt_hp_temp = self.conv_hp2(pt_hp_temp)
            #print(pt_hp_temp.size())
            pt_hp_temp = torch.max(pt_hp_temp, 1)[0].reshape(batch, -1)

            input_rnn[i], input_rnn[i+timestep], input_rnn[i+timestep*2]= pt_sag_temp, pt_cor_temp, pt_ax_temp
            input_rnn_hp[i] = pt_hp_temp

        pt_sag_temp = input_rnn[:timestep,:,:].cuda()
        pt_cor_temp = input_rnn[timestep:timestep*2,:,:].cuda()
        pt_ax_temp = input_rnn[timestep*2:,:,:].cuda()
        
        pt_ax_attn=self.MuA_1(pt_ax_temp,pt_cor_temp,pt_sag_temp)
        pt_cor_attn=self.MuA_2(pt_cor_temp,pt_ax_temp,pt_sag_temp)
        pt_sag_attn=self.MuA_3(pt_sag_temp,pt_cor_temp,pt_ax_temp)
        pt_att_final=self.Lin_img_all(torch.cat((pt_ax_attn, pt_cor_attn,pt_sag_attn),dim=1))
        
        pt_att_final=self.relu(pt_att_final)

        # To here, images along days from (batch, days, embd) to (batch, embd)
        # HPMRI w/wo raw, tumor still (batch, days, emb)
        
        #all_time_attn=all_time_attn.permute(1, 0,2).cuda()

        pt_HPMRI,_ = self.rnn_hp(pt_HPMRI)
        rnn_output_hp,_ = self.rnn_hp_raw(input_rnn_hp.permute(1, 0,2).cuda())
        pt_tumor, _ = self.rnn_t(NMR_data)


        pt_HPMRI, rnn_output_hp, pt_tumor = pt_HPMRI[:,-1,:], rnn_output_hp[:,-1,:], pt_tumor[:,-1,:]
        
        all_rnn_input=torch.cat((pt_att_final,rnn_output_hp,pt_HPMRI,pt_tumor),dim=1)
        
        
        pred=self.final_lin2(all_rnn_input)
        #pred =self.Softmax(pred)
        #pred = self.sigmoid(pred)
        
        return pred 

    def _conv_layer_set_2d(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=( 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool2d((3, 3)),
        )
        return conv_layer

    def _conv_layer_set_3d(self, in_c, out_c, h_size=2):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=( h_size, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((h_size, 3, 3)),
        )
        return conv_layer

    def convs_3d(self,pt_sag_temp):
        
        pt_sag_temp = pt_sag_temp.unsqueeze(1)
        pt_sag_temp=self.conv1_3d(pt_sag_temp)
        pt_sag_temp=self.conv2_3d(pt_sag_temp)
        pt_sag_temp=self.conv3_3d(pt_sag_temp)
        pt_sag_temp=pt_sag_temp.reshape(pt_sag_temp.size(0), self.emb_size, -1).max(-1)[0]
        
        return pt_sag_temp

def train_unit(model,dataloader,class_weights):
    optimizer=Adam(model.parameters(),lr=0.001)
    all_loss=[]
    model.train()
    criterion= nn.CrossEntropyLoss(weight=class_weights.cuda())
    mouse_n, inf = [], []
    for iteration, data_u in enumerate(dataloader):
        HPNRI_data=data_u[0].float() 
        NMR_data=data_u[1].float() 
        NMR_data_mask_1=data_u[2].float() 
        sag_data=data_u[3].float()
        cor_data=data_u[4].float()
        ax_data=data_u[5].float()
        classinf=Variable(data_u[6]).float()
        pt_hp = Variable(data_u[7]).float()

        
        HPNRI_data=HPNRI_data.cuda()
        NMR_data=NMR_data.cuda()
        sag_data=sag_data.cuda()
        cor_data=cor_data.cuda()
        ax_data=ax_data.cuda()
        classinf= classinf.cuda()
        pt_hp = pt_hp.cuda()

        optimizer.zero_grad()
        rnn_output=model(HPNRI_data,sag_data,cor_data,ax_data, pt_hp, NMR_data)
        #rnn_output = torch.softmax(rnn_output)
        #print(rnn_output)
        loss = criterion(rnn_output, classinf.long())
        #loss=criterion(rnn_output.reshape(-1),classinf)

        
        loss.backward()
        optimizer.step()
        all_loss.append(loss.cpu().data.numpy().item())
    
    return all_loss,model

def val_unit(model,dataloader, sv=False):
    all_result_1=[]
    all_gold_1=[]

    all_loss=[]
    model.eval() 

    with torch.no_grad():
        mouse_n, inf = [], []
        for iteration, data_u in enumerate(dataloader):
            HPNRI_data=data_u[0].float() 
            NMR_data=data_u[1].float() 
            NMR_data_mask_1=data_u[2].float() 
            sag_data=data_u[3].float()
            cor_data=data_u[4].float()
            ax_data=data_u[5].float()

            classinf=Variable(data_u[6]).float()
            pt_hp = Variable(data_u[7]).float()

            
            mouse_n = np.concatenate((mouse_n,data_u[8].numpy()))
            mouse_n = np.concatenate((mouse_n,data_u[6].numpy().reshape(-1)))
            
            
            HPNRI_data=HPNRI_data.cuda()
            NMR_data=NMR_data.cuda()
            sag_data=sag_data.cuda()
            cor_data=cor_data.cuda()
            ax_data=ax_data.cuda()
            classinf= classinf.cuda()
            pt_hp = pt_hp.cuda()

            rnn_output=model(HPNRI_data,sag_data,cor_data,ax_data, pt_hp, NMR_data)
            
            pred_ = rnn_output.cpu().detach().numpy().argmax(axis=1)
            mouse_n = np.concatenate((mouse_n,pred_.reshape(-1)))
 
            all_result_1.append(pred_)#rnn_output.cpu().detach().numpy())
            all_gold_1.append(classinf.cpu().detach().numpy())
            

        if sv:
            with open('task2_predictions.csv', 'wb') as f:
                np.savetxt(f, np.array(np.asarray(mouse_n).reshape(-1,3)), delimiter=',', fmt='%d') 
            
    return all_result_1,all_gold_1,mouse_n