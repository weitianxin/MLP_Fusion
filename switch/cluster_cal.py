from sklearn.cluster import KMeans
import pickle
from resmoe_switch import ResmoeSVDSwitchForSequenceClassification as Resmoe,SwitchTransformersConfig, SwitchForSequenceClassification

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


dataset_name = 'mrpc'

switch_config = SwitchTransformersConfig.from_pretrained(
        "google/switch-base-16",
        num_labels=3 if dataset_name == "mnli" else 2, 
        finetuning_task=dataset_name
)

model = SwitchForSequenceClassification.from_pretrained(
            "./switch-base-16",
            config=switch_config
)#.to(device)


def update_ntk_cluster(MLP,layer_idx,expert_idx,is_decoder):
  
        # de = 'decoder' if is_decoder else 'encoder'
        
        # print(f'./extract_saved_16-0/wd-{de}-layer{layer_idx}-ot.pt')
        
        wc = torch.load(f'./extract_saved_16-0/wd-{is_decoder}-layer{layer_idx}-ot.pt').to('cpu')
        
        # print(wc.shape)
        
        # return
        
        # w1 = MLP.wi.weight
        
        
        w1 = torch.cat((MLP.wi.weight,wc[:,:768]),0)        
        
        w2 = torch.cat((MLP.wo.weight,-wc[:,768:].T),1)
        
        print(w1.shape,w2.shape)
        

        cat_w = torch.cat((w1.cpu(),w2.cpu().T),1).detach().numpy()
        # cluster_num = 768
        
        print(cat_w.shape)
        
        torch.save(cat_w, "clustering/{}_{}_{}_{}.pt".format(
                'switch-base-16-resmoe-test', 
                is_decoder, layer_idx, expert_idx))
        
        
        # km = KMeans(n_clusters = cluster_num, n_init = 30, algorithm = "lloyd").fit(cat_w) #n_jobs = 4
        # with open("clustering/{}_{}_{}_{}.pkl".format(
        #         'switch-base-16-resmoe', 
        #         is_decoder, layer_idx, expert_idx), 
        #             'wb') as f:
        #     pickle.dump(km, f)
            
        print(f'{is_decoder}-layer{layer_idx}-{expert_idx}')

        return


        # one-hot clustering matrix
        C = torch.Tensor(km.labels_)
        centers = torch.Tensor(km.cluster_centers_) # d x (2p+1)

        # new parameters for the compressed MLP
        w1_new = centers[:, :768]
        # b1_new = centers[:, self.config.hidden_size]
        w2_new = centers[:, 768:]

        # self.P_count_vec = ScaleLayer(torch.unique(C, return_counts = True)[1])

        # self.wi.weight = Parameter(w1_new)
        # # self.wi.bias = Parameter(b1_new)
        # self.wo.weight = Parameter(w2_new.T)
        
        # self.wi.weight.data = w1_new.to(current_device)
        # # self.wi.bias = Parameter(b1_new)
        # self.wo.weight.data = w2_new.T.to(current_device)
  

def update_ntk_cluster2(MLP,layer_idx,expert_idx,is_decoder):
  
        # de = 'decoder' if is_decoder else 'encoder'
        
        # print(f'./extract_saved_16-0/wd-{de}-layer{layer_idx}-ot.pt')
        
        wc = torch.load(f'./extract_saved_16-0/wd-{is_decoder}-layer{layer_idx}-ot.pt').to('cpu')
        
        # print(wc.shape)
        
        # return
        
        # w1 = MLP.wi.weight
        
        
        w1 = MLP.wi.weight #- wc[:,:768]
        # b1 = self.wi.bias
        # w2 = MLP.wo.weight
        
        w2 = MLP.wo.weight #- wc[:,768:].T
        
        # b1_reshape = torch.unsqueeze(b1,1)
        
        # print(w1.shape,w2.shape)
        
        # return
        
        # current_device = w1.device
        
        # print(torch.cat((w1.cpu(),w2.cpu().T),1).shape, wc.shape, torch.cat((w1.cpu(),w2.cpu().T),1),wc,torch.cat((w1.cpu(),w2.cpu().T),1)-wc)

        cat_w = (torch.cat((w1.cpu(),w2.cpu().T),1) - wc).detach().numpy()
        cluster_num = 768
        
        

        # if self.load_clustering:
        #     with open("clustering/{}_{}_{}_{}.pkl".format(
        #             'switch-base-16', 
        #             de, self.layer_idx, self.expert_idx), 
        #                 'rb') as f:
        #         km = pickle.load(f)
        # else:
        km = KMeans(n_clusters = cluster_num, n_init = 30, algorithm = "lloyd").fit(cat_w) #n_jobs = 4
        with open("clustering/{}_{}_{}_{}.pkl".format(
                'switch-base-16-resmoe2', 
                is_decoder, layer_idx, expert_idx), 
                    'wb') as f:
            pickle.dump(km, f)
            
        print(f'{is_decoder}-layer{layer_idx}-{expert_idx}')

        return  
   
def clusterings(blocks,type,k):
  for i in range(len(blocks)):
    if i<12-k:continue
    if i & 1:
      if type == "encoder":
        mlp = blocks[i].layer[1].mlp.experts
      elif type == "decoder":
        mlp = blocks[i].layer[2].mlp.experts
      expert_indices = mlp.keys()
      for idx in expert_indices:
        update_ntk_cluster(mlp[idx],i,idx,type)
    else:
      continue
 
    
# clusterings(model.transformer.encoder.block,"encoder",10)
# clusterings(model.transformer.decoder.block,"decoder",12)  

clusterings(model.switch_transformers.encoder.block,"encoder",8)
clusterings(model.switch_transformers.decoder.block,"decoder",8)  