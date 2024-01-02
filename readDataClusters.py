#-------------------------------------------------------------------------------
# Name:        module4
# Purpose:
#
# Author:      cagon
#
# Created:     10/01/2023
# Copyright:   (c) cagon 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#!pip install torch
#!pip install torch_geometric
#!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

#import torch.utils.data
import matplotlib.pyplot as plt

import glob
import pandas as pd
import numpy as np
import scipy
import torch
#import torch_geometric
#from torch_geometric.transforms import NormalizeFeatures
#from torch_geometric.data import DataLoader,Data



from pathlib import Path
from typing import List, Tuple
import sys
import pickle as pkl
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from timeit import default_timer
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
os.getcwd()

#C:\Users\cagon\OneDrive\Desktop\Columbia\Data_Jupyter_Download
#C:\Users\cagon\OneDrive\Desktop\GWNETRF\GWNETRF
os.chdir('C:/Users/cagon/OneDrive/Desktop/ClustersGWNETNov2023')

rootdir = 'C:/Users/cagon/OneDrive/Desktop/ClustersGWNETNov2023'

## if scaling

import numpy as np


SCALER = {
    'input_means': np.array([0.0884, 0.542, 0.424, 0.138,0.089, 1.16]),
    'input_stds': np.array([0.359, 1.24, 0.848, 0.387, 0.813, 2.87]),
    'output_mean': np.array([0.103]),
    'output_std': np.array([0.475])
}


def toTensor(nparr, dtype):
    """utility function for tensor conversion
    """
    return torch.tensor(nparr, dtype=dtype)

import pandas as pd
rootdir = 'C:/Users/cagon/OneDrive/Desktop/ClustersGWNETNov2023'


def _read_info(path):
    #modified from
    #https://github.com/kratzert/lstm_for_pub/blob/master/extract_benchmarks.py

    _info = pd.read_csv(path)
    
    _info.columns=['Cluster','lat','lng']
    


    _info['gauge_str'] = _info['Cluster']


    _info['Cluster'] = _info['Cluster'].apply(pd.to_numeric)
    _info['lat'] = _info['lat'].apply(pd.to_numeric)
    _info['lng'] = _info['lng'].apply(pd.to_numeric)

    return _info

def getStaticAttr():

    zonelistfile = f'{rootdir}/ClusterList.txt'
    df_zoneset = pd.read_csv(zonelistfile, header=None)
    df_zoneset.columns=['Cluster']


    meta_df = _read_info(f'{rootdir}/ClusterInformation.csv')
    assert meta_df['Cluster'].is_unique  
    meta_df = meta_df.round({
        'lat': 5,
        'lng': 5
    })  # latitudes and longitudes should be to 5 significant digit



    meta_df = df_zoneset.join(
            meta_df.set_index('Cluster'),
            on='Cluster')


    # load characteristics file (with areas)
    rootloc = f'{rootdir}/'  # catchment characteristics file name

    char_df = pd.read_excel('StaticAttributesCluster.xlsx', dtype={'Cluster': int}) 
    static_df = char_df
    

    print(char_df)
    nzones = static_df.shape[0]  # count number of zones

    #print(meta_df)
    print ('number of zones', nzones)
    print('static list', static_df)

    return static_df

def getSubSet(allDF):
    """Return a subset of static attribute dataframe
    Reference: Nearing 2019 WRR paper, Table 1
    """

    #RES04	EMP06	COM02	HTEN02	MORT02	Slope	Elevation	lat	lng	Area	PopDens	Imperv	Buildings	Footprint	FootperSq
    #colnames = ['RES04',	'EMP06',	'COM02',	'HTEN02',	'MORT02',	'lat',	'lng',	'PopDens']
    colnames = ['Slope','Elevation','Area',	'Imperv','BLD','FTP','lat',	'lng']
    return  allDF[colnames]

def getSubSet4Clustering(allDF):
    """Return a subset of static attribute dataframe for clustering
    Reference: Nearing 2019 WRR paper, Table 1
    """

    colnames = ['Slope','Elevation','Area',	'Imperv','BLD','FTP','lat',	'lng']
    #colnames = ['RES04',	'EMP06',	'COM02',	'HTEN02',	'MORT02',	'lat',	'lng',	'PopDens']
    return  allDF[colnames]

from pathlib import Path, Path
from typing import List, Tuple
import pandas as pd

def load_forcing(Data_root: Path, forcingType: str, zone: str) -> Tuple[pd.DataFrame, int]:

    Data_root = Path('C:/Users/cagon/OneDrive/Desktop/ClustersGWNETNov2023')
    forcing_path = Data_root / 'PredictorsClusterTopoX6'


    files = list(forcing_path.glob("*.xlsx"))
    #print(files)
    #print(files)
    #file_path = [f for f in files if int(float(f.stem[:3])) == zone]
    file_path = [f for f in files if int(float(f.stem)) == zone]

    if len(file_path) == 0:
        raise RuntimeError(f'No file for zone {zone} at {file_path}')
    else:
        file_path = file_path[0]
    
    
    print(file_path)
    #col_names = ['date', 'Catch', 'PRCP']
    df = pd.read_excel(file_path)
    #df = pd.read_excel(file_path, usecols="A,C,F", names=col_names)
    #print(df)
    #df = pd.read_excel(file_path, names=col_names, converters={'date': lambda dt: pd.to_datetime(dt, format='%d%m%Y', errors='coerce')})
    
    #df.columns = df.columns.str.strip()
    #df.index = df[df.columns[0]]
    
    dates = (df[df.columns[0]].map(str))

    print(dates)
    df.index = pd.to_datetime(dates)


    # load area from header
    ###with open(file_path, 'r') as fp:
        ###content = fp.readlines()
        ###area = int(content[2])

    return df ###,area

def load_response(Data_root: Path, zone: str) -> pd.Series:


    Data_root = Path('C:/Users/cagon/OneDrive/Desktop/ClustersGWNETNov2023')
    response_path = Data_root / 'ResponseClusterTopoX6'
    
    files = list(response_path.glob('*.xlsx'))
    file_path = [f for f in files if int(float(f.stem)) == zone]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for zone {zone} at {file_path}')
    else:
        file_path = file_path[0]
    
    #col_names = ['date', 'QObs']
    #df = pd.read_excel(file_path, names=col_names)
    df = pd.read_excel(file_path)
    ###df.index = pd.to_datetime(df['date'], format="%Y/%m/%d")
    
    df.columns = df.columns.str.strip()
    df.index = df.Date
    dates = (df.Date.map(str))

    df.index = pd.to_datetime(dates)
    print(df.Flooding)



    return df.Flooding

import numpy as np

def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    #https://github.com/kratzert/lstm_for_pub/blob/bb74c3ff3047d2f60a839fa05a4e621587225205/papercode/datautils.py

    #https://github.com/kratzert/lstm_for_pub/blob/bb74c3ff3047d2f60a839fa05a4e621587225205/papercode/datautils.py
    """Normalize features 
    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs  and `output` that the `feature` input are response
        values.
    Returns
    -------
    np.ndarray
        Normalized features
    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """

    return feature




def new_normalize_features(feature: np.ndarray, variable: str, SCALER: dict) -> np.ndarray:
    """normalize features according to the training period
    [note this should be used instead of the hard-coded scaler]
    Parameters
    ----------
    feature: input/output numpy array
    variable: either "input" or "output
    scaler: the scaler to be used in standard scaling
    Returns
    -------
    feature: scaled feature
    """

    return feature

    #if variable == "inputs":
        #feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    #elif variable == 'output':
     #   feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    #else:
     #   raise RuntimeError(f"Unknown variable type {variable}")
    #return feature


def genLSTMData(rootfolder, zoneList, dates, seq=6, reGen=False, genDF=False, **kwargs):
    """Generate input for graph
    Parameters
    ---------
    rootfolder: posixpath, root of Data dataset
    zoneList: string list of all zones
    dates: the start and end dates of training and testing data (note training includes train/val periods)
    seq: lookback period
    reGen, true to regenerate datasets
    genDF: true to reload Data forcing DF
    """
    dir_path = os.path.dirname(os.path.realpath("__file__"))

    #get training and testing dates
    train_start,train_end = dates[0:2]
    test_start,test_end = dates[2:]
    #set default values
    kwargs.setdefault('includeStatics', True)
    kwargs.setdefault('staticdf', None)
    kwargs.setdefault('forcingType', 'PredictorsClusterTopoX6')###'nldas')

    addStatics = kwargs['includeStatics']
    dfStatics = kwargs['staticdf']
    forcingType = kwargs['forcingType']
    latentType = kwargs['latentType']

# Forcing (time series data)

    if addStatics:
        assert(not dfStatics is None)
    if addStatics:
        saved_file = os.path.join(dir_path,f'Data_lstmlist_{forcingType}_{latentType}_seq{seq}_static_new.pkl')
    else:
        saved_file = os.path.join(dir_path,f'Data_lstmlist_{forcingType}_seq{seq}_new.pkl')

    if reGen:
        if genDF:
            allDF=[]
            allQ =[]
            #this loop puts all data in lists
            for zone in zoneList:
                print (zone)
                df = load_forcing(rootfolder, forcingType, zone=zone)
                dfQ = load_response(rootfolder, zone=zone)
                
                #forcing var's to use in this project
                #colnames = ['WB', 'SB','CB', 'PRCP', 'SNOW','MaxValue_Hourly'] #SB	CB	PRCP	SNOW
                colnames = ['SB',	'CB',	'PRCP',	'SNOW',	'MaxValue_Hourly']

                #['WB', 'SB', 'PRCP', 'SNOW', 'MaxValue_Hourly']
                #subsetting
                
                df = df[colnames]
                #df = df[['WB', 'SB','CB', 'PRCP', 'SNOW','MaxValue_Hourly']].copy()
                df = df[['SB',	'CB',	'PRCP',	'SNOW',	'MaxValue_Hourly']].copy()
                #this makes dfQ index the same as forcing data index
                
                dfQ = dfQ.reindex(df.index)

                #subsetting on time [use trainstart-seq+1 to be consistent with Krazert]
                #df   = df[train_start-pd.Timedelta(seq-1, unit='D'):test_end]
                #dfQ = dfQ[train_start-pd.Timedelta(seq-1, unit='D'):test_end]
                df   = df[train_start:test_end]
                dfQ = dfQ[train_start:test_end]
                
                
                #print('forcing dataframe', df)
                #print('response dataframe', dfQ)
                #make sure the two dataframes have the same length
                assert(df.shape[0]==dfQ.shape[0])
                allDF.append(df)
                allQ.append(dfQ)
                #for debugging
                """
                if zone in ['05120500']:
                    plt.figure()
                    plt.plot(dfQ[test_start:test_end])
                    plt.savefig('Q{0}.png'.format(zone))
                    plt.close()
                    sys.exit()
                """
            # do normalization
            #get mean correspoinding to training
            inputDF=[]
            outputDF=[]
            #get training data for all zones
            for df1,df2 in zip(allDF,allQ):
                inputDF.append(df1[:train_end])
                outputDF.append(df2[:train_end])
            #calculate stats
            bigDF = pd.concat(inputDF, axis=0)
            bigQDF = pd.concat(outputDF,axis=0)
            bigQMat = bigQDF.to_numpy()
            #do this to remove invalid values in Q
            #print(allDF)
            #bigQMat[bigQMat<0.0]=np.NaN
            input_means = np.nanmean(bigDF.to_numpy(), axis=0)
            input_stds = np.nanstd(bigDF.to_numpy(), axis=0)
            output_mean = np.nanmean(bigQMat, axis=0)
            output_std = np.nanstd(bigQMat, axis=0)
            myscaler = {
                'input_means': input_means,
                'input_stds': input_stds,
                'output_mean': output_mean,
                'output_std': output_std,
            }
            ####print ('training scaler', myscaler)
            bigDF=None
            bigQDF=None
            bigQMat=None

            for izone in range(len(allDF)):
                arr = new_normalize_features(allDF[izone].to_numpy(),'inputs',myscaler)
                df = pd.DataFrame(arr,index=allDF[izone].index)
                #df.columns = colnames
                df.columns = colnames
                allDF[izone]=df


                arr = allQ[izone].to_numpy()
                #arr[arr<0.0] = np.NaN
                arr = new_normalize_features(arr, 'output', myscaler)
                dfQ = pd.DataFrame(arr,index=allQ[izone].index)
                allQ[izone] = dfQ
                '''
                test location 05120500
                if izone == 288:
                    plt.figure()
                    plt.plot(dfQ[test_start:test_end])
                    plt.savefig(f'Q{izone}.png')
                    plt.close()
                    sys.exit()
                '''

            pkl.dump(allDF, open(f'Data_forcingdf_{forcingType}_{latentType}_new.pkl', 'wb'))
            pkl.dump(allQ, open('Data_flow_new.pkl', 'wb'))
            pkl.dump(myscaler, open(f'Data_forcing_scaler_{forcingType}.pkl', 'wb'))

        else:
            allDF=pkl.load(open(f'Data_forcingdf_{forcingType}_{latentType}_new.pkl', 'rb'))
            allQ =pkl.load(open('Data_flow_new.pkl', 'rb'))

        #assumption, the forcing data is contiguous
        nzones = len(allDF)
        assert(nzones==len(zoneList))

        starttime= default_timer()

        #join all dataframes
        bigQDF = pd.concat(allQ,axis=1)
        ndays = bigQDF.shape[0]
        print(ndays)
        
        ###trainLen = len(pd.date_range(train_start-pd.Timedelta(seq-1, unit='D'), train_end, freq='D'))
        #trainLen = len(pd.bdate_range(train_start, train_end))
        ###testLen =  ndays-trainLen
        #testLen = ndays - trainLen - 1
        #trainLen = len(pd.bdate_range(train_start, train_end))
        #testLen = len(pd.bdate_range(test_start, test_end))
        trainLen = len(pd.date_range(train_start, train_end)) +1
        testLen = len(pd.date_range(test_start, test_end)) + 1



       
        #the following should be 1096 days
        
        
        print ('test', len(pd.date_range(test_start, test_end, freq='D')))
        print ('all days=', ndays, 'train days ', trainLen, 'test days', testLen)
        print ('finished getting dataframes ...')
        print ('preparing graph data ...')
        """this should print the same figure as the previous Q{izone}.png
        plt.figure()
        plt.plot(bigQDF.loc[test_start:test_end].iloc[:,288])
        plt.savefig('big_Q288.png')
        plt.close()
        sys.exit()
        """
# Static Data

        nfeature_d = allDF[0].shape[1]
        nfeatures=nfeature_d
        if addStatics:
            nfeature_s = dfStatics.shape[1] #number of static features
            nfeatures += nfeature_s
        Xtrain_list = []
        ytrain_list = []
        Xtest_list = []
        ytest_list = []

        for irow in range(seq, ndays):
            targetvec = bigQDF.iloc[irow-1,:].to_numpy()

            if not np.isnan(targetvec).any():
                #assuming all data in forcing is continuous
                featureMat = np.zeros((1,nzones,nfeatures))
                for izone in range(nzones):
                    featureMat[:,izone,:nfeature_d] = allDF[izone].iloc[irow-1:irow,:]
                    if addStatics:
                        featureMat[:,izone,nfeature_d:]= dfStatics.iloc[izone,:].to_numpy()[np.newaxis,:].repeat(1,axis=0)

                if irow<trainLen:
                    Xtrain_list.append(featureMat)
                    ytrain_list.append(targetvec)

                else:
                    Xtest_list.append(featureMat)
                    ytest_list.append(targetvec)

        ####print ("time taken ", default_timer()-starttime)
        ####print (f'# train: {len(ytrain_list)}, # test: {len(ytest_list)}')
        ####print(ytrain_list)
        ####print(ytest_list)
        pkl.dump([Xtrain_list,ytrain_list,Xtest_list,ytest_list], open(saved_file, 'wb'))
    else:
        Xtrain_list,ytrain_list,Xtest_list,ytest_list = pkl.load(open(saved_file, 'rb'))
        
        #print(ytrain_list)
        #print(ytest_list)
        #print(Xtrain_list)
        #print(Xtest_list)

    return Xtrain_list,ytrain_list,Xtest_list,ytest_list

def genLSTMDataSets(forcingType, latentType, splitratio=(0.8,0.2), seq=6, addStatics=True):
    """Generate LSTM Datasets for training, validation and testing

    Parameters
    ---------
    forcingType: type of forcing datasets
    splitratio: (train,val)
    seq: sequence length
    addStatics: True to include static attributes

    Returns
    --------
    train, validation, and testing datasets
    """

    from torch.utils.data import DataLoader, TensorDataset
    #from torch.utils.data import DataLoader
    import os.path
    dir_path = os.path.dirname(os.path.realpath("__file__"))
    if addStatics:
        #datafile = os.path.join(dir_path, f'Data_lstmlist_{forcingType}_seq{seq}_new.pkl')
        datafile = os.path.join(dir_path, f'Data_lstmlist_{forcingType}_{latentType}_seq{seq}_static_new.pkl')
    else:
        datafile = os.path.join(dir_path, f'Data_lstmlist_{forcingType}_seq{seq}_new.pkl')
    ####print ('use data from ', datafile)
    Xtrain_list,ytrain_list,Xtest_list,ytest_list = pkl.load(open(datafile, 'rb'))

    nData = len(Xtrain_list)
    nTrain = int(nData*splitratio[0])
    nVal = int(nData*splitratio[1])
    #training
    Xin = toTensor(np.asarray(Xtrain_list[:nTrain]),dtype=torch.float32)
    y  =  toTensor(np.asarray(ytrain_list[:nTrain]),dtype=torch.float32)
    print ('train data', Xin.shape, y.shape)
    trainDataset = TensorDataset(Xin,y)
    print(trainDataset)
    #validation
    Xin = toTensor(np.asarray(Xtrain_list[nTrain:]),dtype=torch.float32)
    y  =  toTensor(np.asarray(ytrain_list[nTrain:]),dtype=torch.float32)
    valDataset = TensorDataset(Xin,y)
    print ('val data', Xin.shape, y.shape)
    #testing
    #Xin = toTensor(np.asarray(Xtest_list),dtype=torch.float32)
    #y  =  toTensor(np.asarray(ytest_list),dtype=torch.float32)

    #testDataset = TensorDataset(Xin,y)
    #print ('test data', Xin.shape, y.shape)
    #nfeatures=Xin.shape[-1]
    
    #testing
    ##Xin = toTensor(np.asarray(Xtest_list), dtype=torch.float32)
    ##y = toTensor(np.asarray(ytest_list), dtype=torch.float32)
    Xin = toTensor(np.asarray(Xtest_list), dtype=torch.float32)
    y = toTensor(np.asarray(ytest_list), dtype=torch.float32)
    testDataset = TensorDataset(Xin, y)
    print('test data', Xin.shape, y.shape)
    nfeatures = Xin.shape[-1]

    return trainDataset,valDataset,testDataset,nfeatures

def loadGraphWeight(filename=None):
    print ('use weight matrix', filename)
    D = scipy.sparse.load_npz(filename)
    return torch_geometric.utils.from_scipy_sparse_matrix(D)

import os

def main():
    #training period and test period
    #based on hXtest_listttps://github.com/kratzert/lstm_for_pub/blob/master/main.py
    #i modified the train_end to significantly increase training data sizes
    train_start = pd.to_datetime('2010-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    train_end = pd.to_datetime('2018-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    test_start = pd.to_datetime('2018-01-02 00:00:00', format='%Y-%m-%d %H:%M:%S')
    test_end = pd.to_datetime('2019-12-31 00:00:00', format='%Y-%m-%d %H:%M:%S')

    camelDates = [ train_start, train_end, test_start, test_end]
    dfStaticAll = getStaticAttr()
    df = getSubSet(dfStaticAll)
    #df = getSubSet4Clustering(dfStaticAll)
    rootfolder = Path('C:/Users/cagon/OneDrive/Desktop/ClustersGWNETNov2023')
    #
    #
    #for lstm, setting the parameters
    #>>>>>>
    seq=6
    latentdim=6
    addstatics=True
    loadLatent=False
    forcingType= 'PredictorsClusterTopoX6' ###
    if loadLatent:
        latentType=str(latentdim)
    else:
        latentType='full'
    #<<<<<<
    if loadLatent:
        print ("use latent static matrix")
        latentMat = np.load(f'latentMat_dim{latentdim}.npy')
        #normalize the static features
        df = pd.DataFrame(StandardScaler().fit_transform(latentMat),index=df.index)
    else:
        print ('use full static attribute matrix')
        df = pd.DataFrame(StandardScaler().fit_transform(df.to_numpy()),index=df.index)
    print ('static attribute matrix', df.shape)


    kwargs={'includeStatics':addstatics,
            'staticdf':df,
            'latentType':latentType,
            'forcingType': forcingType,
            }

    Xtrain_list,ytrain_list,Xtest_list,ytest_list =  genLSTMData(rootfolder,
            dfStaticAll['Cluster'],
            seq=seq,
            reGen=True,
            dates=camelDates,
            genDF=True,
            **kwargs
            )

    trainDataset,valDataset,testDataset,_ = genLSTMDataSets(forcingType, latentType, seq=seq,addStatics=addstatics)

if __name__ == '__main__':
    main()

