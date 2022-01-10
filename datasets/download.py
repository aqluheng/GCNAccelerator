from torch_geometric.datasets import Planetoid, NELL, Reddit
CoraDataset = Planetoid("./Cora","Cora")
CiteSeerDataset = Planetoid("./CiteSeer","CiteSeer")
PubMedDataset = Planetoid("./PubMed","PubMed")
NellDataset = NELL("./Nell")
RedditDataset = Reddit("./Reddit")