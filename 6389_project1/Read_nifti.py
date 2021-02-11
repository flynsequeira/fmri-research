import nibabel as nib # need to install nibabel package

# nifti_file is .nii.gz file
def read_nifti_file(nifti_file):
    nii_image = nib.load(nifti_file)
    nii_data = nii_image.get_data()
    return nii_data
