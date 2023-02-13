
import argparse
import SimpleITK as sitk
import numpy as np
import os, shutil, sys, getopt

parser = argparse.ArgumentParser(description="Ensemble")
parser.add_argument("--path", default="/tmp", type=str, help="training path")
parser.add_argument("--class_id", default=1, type=int, help="class_id")
parser.add_argument("--ngroups", default=10, type=int, help="number of groups")

global opt
opt = parser.parse_args()

#####################################################################################################
class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:  
                    if  special_file.endswith(self.postfix):    
                        files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    if special_file.startswith(self.prefix):  
                        files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list      

#####################################################################################################
def main():
    opt=parser.parse_args()
    print(opt)
    path=opt.path
    num_groups = opt.ngroups
    class_id = opt.class_id

    postfix_tmp = 'class' + str(class_id) +'.nii.gz'
    print(postfix_tmp)    
    scan_files = ScanFile(path, postfix=postfix_tmp)  
    filenames = scan_files.scan_files()  

    for filename in filenames:         

        seg_fn = filename
        seg = sitk.ReadImage(seg_fn)
        segnp = sitk.GetArrayFromImage(seg)

        spacing = seg.GetSpacing()
        origin = seg.GetOrigin()
        direction = seg.GetDirection()

        file_out = filename.replace('/group_1/', '/')
        print(file_out)
        #print(seg_fn)
        #print(filename)

        for i in range(2, num_groups+1):
            str_group = 'group_' + str(i)
            seg_fn = filename.replace('group_1', str_group)
            #print(seg_fn)

            seg_tmp = sitk.ReadImage(seg_fn)
            seg_tmp = sitk.GetArrayFromImage(seg_tmp)

            segnp = segnp + seg_tmp

        segnp = segnp / num_groups

        if class_id==1:
            segnp[segnp<0.5]=0
            segnp[segnp>=0.5]=1

        if class_id==2:
            segnp[segnp<0.3]=0
            segnp[segnp>=0.3]=1

        #segnp.astype(np.float)
        volout = sitk.GetImageFromArray(segnp)
        volout.SetSpacing(spacing)
        volout.SetOrigin(origin)
        volout.SetDirection(direction)
        sitk.WriteImage(volout, file_out)

if __name__ == '__main__':
    main()
