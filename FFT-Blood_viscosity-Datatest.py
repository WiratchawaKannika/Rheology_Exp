import json 
import pandas as pd
import numpy as np
from pathlib import Path
import PIL
from PIL import Image
import cv2 
import tqdm
import os
import glob
import ast
import argparse

# set number of CPUs to run on
ncore = "12"
# set env variables
# have to set these before importing numpy
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore

## Function: Convert image in 2D fast fourier transform signal
def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def Convert2Dfft(img_path1, img_path2):
    img1 =  cv2.imread(img_path1).astype(np.float64)  # BGR, float
    #print(img1.shape)
    img2 =  cv2.imread(img_path2).astype(np.float64)  # BGR, float
    #print(img1.shape)
    img_green = np.absolute(img2[:, :, 1] - img1[:, :, 1])  # g = |g2 - g1|
    _img_green = img_green.astype(np.uint8)  # convert back to uint8
    ft = calculate_2dft(_img_green)
    ft2 = np.power(abs(ft), 2)   ## เก็บค่านี้ 
    return ft2




def main():
    my_parser = argparse.ArgumentParser(description='Convert Images to 2dFFT images; MSD')
    my_parser.add_argument('--pth2dt', type=str, help='path/to/filename.json', default='/media/SSD/rheology2023/P0100_D0_30HZ_20XINF_UWELL_20221222_174014_MSDT.json')
    my_parser.add_argument('--pth2dataset', type=str, help='Rheology Blood Viscosity Dataset', default='/home/kannika/codes_AI/Rheology_Blood/Dataset_Rheology_Blood_Viscosity_HN_NBL-Dataset-6Fold.csv')
    my_parser.add_argument('--pth2save', type=str, help='Root pth to save 2dfft DATASET', default='/media/HDD/rheology2023/Blood_Viscosity_2dFFT/2dFFT_dataset/SSD_Backup_3')
    my_parser.add_argument('--pth2saveCSV', type=str, help='pth to save DataFrame 2dfft DATASET', default='/home/kannika/codes_AI/Rheology_Blood')

    args = my_parser.parse_args()

    ## Get parse args
    #pth2dt = args.pth2dt
    #pth2dataset = args.pth2dataset
    pth2save = args.pth2save
    pth2saveCSV = args.pth2saveCSV
    
    ''' 
    Read json Stamp time 
    '''
    json_path = args.pth2dt    ## Fixed 
    f = open (json_path, "r")
    # Reading from file
    data = json.loads(f.read())
    dfMSDT = pd.DataFrame(data)
    dfMSDT = dfMSDT[dfMSDT["dt"]<=10].reset_index(drop=True)
    print(f"MSDT .json : {dfMSDT.shape}")
    
    ### *Step 2 ==> import data folder : Glycerol
    image_name_, image_path_, image_folder_, folder_source_, classes_, Code_, subclass_, fold_ = [],[],[],[],[],[],[],[]
    Dataset = pd.read_csv(args.pth2dataset) ## Config
    
    ### *Step 3 ==> Prepare images data  <=== 
    datafolder = list(set(Dataset['image_folder']))
    datafolder.sort()
    print(f"Found Validate Dataset: {len(datafolder)} Folder ==> {Dataset.shape[0]} images")
    ##*** **********************************************************************************************************************
#     lst_FlderName = []
#     baseRoot = "/media/HDD/rheology2023/Blood_Viscosity/2dFFT_dataset/SSD_Backup"
#     getdir = glob.glob(f"{baseRoot}/*")
#     getdir.sort()
#     for j in range(len(getdir)):
#         FlderName = getdir[j].split("/")[-1]
#         lst_FlderName.append(FlderName)
    
#     '''
#     subtract list Folder Name:
#     '''
#     set_foder = list(set(datafolder)-set(lst_FlderName))
#     print(f"Found Validate Residual: {len(set_foder)} Folder")
   
    for k in range(len(datafolder)):
#     for k in range(len(set_foder)):
        DFfolder = Dataset[Dataset["image_folder"]==datafolder[k]].reset_index(drop=True)
        #DFfolder = Dataset[Dataset["image_folder"]==set_foder[k]].reset_index(drop=True)
        #print(DFfolder.shape)
        #DFfolder.head()
        img_path = DFfolder["image_path"].tolist()
        img_path.sort()

        ## Set element to save as DataFrame
        folder_source = list(set(DFfolder["folder_source"]))
        folder_source_i = folder_source[0]
        classes = list(set(DFfolder["classes"]))
        classes_i = classes[0]
        Code = list(set(DFfolder["Code"]))
        Code_i = Code[0]
        subclass = list(set(DFfolder["subclass"]))
        subclass_i = subclass[0]
        fold = list(set(DFfolder["fold"]))
        fold_i = fold[0]

        '''
        Run 2dFFT 
        '''

        for i in range(len(dfMSDT)):
            avgFFT_dt = list()  ## array == 20 ids 
            idt = dfMSDT['dt'][i] ## number idt 
            list_initTime = dfMSDT['it'][i]   ##a list 
            for t in list_initTime:
                #print(t)
                ## 1. Init images
                init_img = t-1
                img_path1 = img_path[init_img]
                ## 2. idt images
                idt_img = init_img+idt
                img_path2 = img_path[idt_img]
                FFT = Convert2Dfft(img_path1, img_path2)  # Use function 
                avgFFT_dt.append(FFT)
            DDM_dt = np.sum(avgFFT_dt, axis=0)/len(avgFFT_dt)
            ## NUMPY SAVE and FFT images save 
            base_folder = datafolder[k]
            #savepth = datafolder.replace('Glycerol', FolName2Save)
            mkdir_pthFFT = f"{pth2save}/{base_folder}"
            mkdir_pthnumpy = mkdir_pthFFT.replace('2dFFT_dataset', "npyfiles_dataset")
            ##---** Create Folder to Save 
            import imageio
            os.makedirs(mkdir_pthnumpy, exist_ok=True)
            os.makedirs(mkdir_pthFFT, exist_ok=True)
            ### Create Name to save
            numpy_name = mkdir_pthnumpy+'/'+base_folder+'_idt'+str(idt)+'.npy'
            print(f"Save Numpy images as ==> [{numpy_name}]") 
            img_name = mkdir_pthFFT+'/'+base_folder+'_idt'+str(idt)+'.png'
            print(f"Save FFT images as ==> [{img_name}]") 
            ##-- NUMPY SAVE
            np.save(numpy_name, DDM_dt)
            ## images SAVE to .png
            output_image = np.log(DDM_dt)
            imageio.imwrite(img_name, output_image) 
            print("="*125)
            ## Createa Data 
            image_name_.append(base_folder+'_idt'+str(idt)+'.png') 
            image_path_.append(img_name)  
            image_folder_.append(base_folder)  
            folder_source_.append(folder_source_i)  
            classes_.append(classes_i)  
            Code_.append(Code_i)  
            subclass_.append(subclass_i)  
            fold_.append(fold_i)   
        print(f"Prepare 2dFFT Images : ===> DONE!!  <===")
    ## Create Data Frame
    dict = {'image_name': image_name_, 'image_path': image_path_, 'image_folder': image_folder_, 'folder_source':folder_source_,
             'classes':classes_, 'Code_':Code_, 'subclass_':subclass_, 'fold':fold_}
    df_2dFFT = pd.DataFrame(dict)
    print(f"[INFO]: 2dFFT Dataset : Blood Viscosity ==> With Shape {df_2dFFT.shape}")
    pth2saveCSV_ = f"{pth2saveCSV}/Dataset_Rheology_Blood_Viscosity_HN_NBL-2dFFTdataset-6Fold.csv"  ##** Set Csv Name. 
    df_2dFFT.to_csv(pth2saveCSV_)
    print(f"Save Dataframe : DONE!!! at {pth2saveCSV_}")


## Run Function 
if __name__ == '__main__':
    main()
    
    
    



