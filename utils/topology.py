import numpy as np
from tqdm import tqdm
import tarjan

class topology():
    seg = None
    idx_list = None
    idx_mat = None
    adjList = None
    pair_list = None
    stage = None
    
    def __init__(self, seg):
        self.seg = seg
        tmp, counts = np.unique(self.seg, return_counts=True)
        self.idx_list = [tmp[i] for i in np.argsort(tmp)[::-1] 
                    if (counts[i] / self.seg.size * 100) > 0.1]
        # TODO: OPTIMIZE HERE! USE np.apply_over_axis() please:)
        self.init_adjmatrix()
        self.init_stage()
        
    def init_adjmatrix(self):
        self.idx_mat = [ [0 for j in self.idx_list] for i in self.idx_list]
        try:
            with tqdm(range(5, self.seg.shape[0] - 5), leave=False) as t:
                for x_i in t:
                    for y_i in range(5, self.seg.shape[1] - 5):
                        tmp, counts = np.unique(self.seg[x_i - 1 : x_i + 1, y_i - 1: y_i + 1], return_counts=True)
                        for i in range(len(tmp)):
                            if tmp[i] not in self.idx_list or self.seg[x_i][y_i] not in self.idx_list: continue
                            self.idx_mat[self.idx_list.index(self.seg[x_i][y_i])][self.idx_list.index(tmp[i])] += 1
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        self.idx_mat = np.array(self.idx_mat)
        
    def init_stage(self, stage=None):
        """
        Initialize the stages.
        """
        if stage is not None:
            self.stage = stage
            return self.stage
        if self.pair_list is None: self.convert2edgelist()
        self.stage = [np.array(list(set(self.pair_list[:, 0]) - set(self.pair_list[:, 1])),)] # stage 0
        while(not self.check_stage_wholeness(self.stage) and self.stage[-1].shape[0]):
        #     self.stage.append(np.array(
        #         list(set(
        #             self.pair_list[:, :][np.where(np.in1d(self.pair_list[:, 0], self.stage[-1]))][:, 1]
        #         ))
        #     ))
            tmp_final_stage = self.stage[-1]
            self.stage, flg = self.check_current_final_stage_element(self.stage)
            if flg is False:
                break
        if not self.check_stage_wholeness(self.stage):
            stage_set = set([i for j in self.stage for i in j])
            idx_set = set(self.idx_list)
            self.stage.append(np.array(list(idx_set - stage_set)))
        return self.stage
        
        
    def convert2adjlist(self, adjmatrix=None):
        """
        Convert adjacency matrix to adjacency list 
        """
        if adjmatrix is not None: self.idx_mat = adjmatrix
        self.adjList = {self.idx_list[i] : [] for i in range(self.idx_mat.shape[0])}
        for i in range(self.idx_mat.shape[0]): 
            for j in range(self.idx_mat.shape[1]): 
                if self.idx_mat[i][j] > 0 and i != j:
                    if self.idx_list[j] == -1:
                        self.adjList[self.idx_list[j]].append(self.idx_list[i]) 
                    else:
                        self.adjList[self.idx_list[i]].append(self.idx_list[j]) 
        return self.adjList
    
    def convert2edgelist(self, adjmatrix=None):
        if adjmatrix is not None: self.idx_mat = adjmatrix
        pair = np.array(np.where((self.idx_mat - self.idx_mat.T)[:, :] > 0))
        self.pair_list = []
        for i in range(len(pair[0])):
            if self.idx_list[pair[1][i]] == -1:
                self.pair_list.append([self.idx_list[pair[1][i]], self.idx_list[pair[0][i]]])
            else:
                self.pair_list.append([self.idx_list[pair[0][i]], self.idx_list[pair[1][i]]])
        self.pair_list = np.array(self.pair_list)
    
    def circles(self, adjlist=None):
        """
        cannot remove them for they are too complex:p
        so just find them.
        TODO: remove the circles!!!
        """
        if adjlist is not None:
            return tarjan.tarjan(adjlist)
        if self.adjList is None: self.convert2adjlist()
        return tarjan.tarjan(self.adjList)
    
    def check_stage_wholeness(self, stage, idx_list=None):
        """
        Check the wholeness of our stage.
        return:
            True - whole
            False - incomplete
        """
        if idx_list is None: idx_list = self.idx_list
        stage_set = set([i for j in stage for i in j])
        idx_set = set(idx_list)
        return (len(idx_set - stage_set) == 0)
    
    def check_current_final_stage_element(self, stage, pair_list=None):
        """
        Check whether the elements in all stages are the subelements of the final stages.
        TODO: Need to optimize
        return:
            stage - result stage
            flg :
                True - have been reset
                False - no problem
        """
        if pair_list is None: pair_list = self.pair_list
        tmp_new = np.array(list(pair_list[:, :][np.where(np.in1d(pair_list[:, 0], stage[-1]))][:, 1]))
        flg_new = np.in1d(tmp_new, tmp_new) # elements should be added
        for i in range(len(stage[:])):
    #         flg = np.in1d(stage[i], tmp_new)
            flg_new[np.in1d(tmp_new, stage[i])] = False
            if np.any(flg_new): continue
    #             print("bomb.")
    #             tmp_remain = stage[i][np.where(~flg)]
    # #             stage = stage[:i]
    # #             stage.append(tmp_remain)
    #             stage[i] = tmp_remain
    #             stage.append(np.array(list(set(tmp_new))))
    # #             return stage, True
        tmp_new_remain = tmp_new[np.where(flg_new)]
        if tmp_new_remain.shape[0] == 0: return stage, False
        stage.append(np.array(list(set(tmp_new_remain))))
        return stage, True
    
    def display_stage(self, names, stage=None):
        """
        Display the stage in name format:
        input:
            names - {idx : names}
            stage
        return:
            [names]
        """
        if stage is None: stage = self.stage
        return [{names[i + 1] for i in j} for j in stage]
    
    def combine_seg_result(self, seg_re, stage=None):
        if stage is None: stage = self.stage
        return [{i: seg_re[i] for i in j} for j in stage]