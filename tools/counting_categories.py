import glob
import json
def count_cat(train_file,cate_dict,output_file):
	# this function takes all the images in train_file and look for the annotations to 
		# to count the number of categories in all the images.
	# Inputs:
	# =======
	# train_file (str) file with the directions of the images (note that in the annotations must be in the same path of the image)
	# cate_dict (dict) cate_dict[i] is the name of the i-th category 
	# output_file (str) outputfile
	# Output:
	# =======
	# dictionary with the number of elements in every category
	f=open(train_file,"r")
	list_images=f.read().splitlines()
	f.close()
	num_in_cate={}
	for i in list_images:
		i=i.replace(".jpg",".txt")
		f=open(i,"r")
		ann_in_image=f.read().splitlines()
		f.close()
		for j in ann_in_image:
			print(j)
			if cate_dict[int(j[0])] in list(num_in_cate.keys()):
				num_in_cate[cate_dict[int(j[0])]]+=1
			else:
				num_in_cate[cate_dict[int(j[0])]]=1
	print(num_in_cate)
	out=open(output_file,"w")			
	json.dump(num_in_cate,out)
	out.close()

if __name__ == '__main__':
	# train_file (str) file with the directions of the images (note that in the annotations must be in the same path of the image) 
	# diff_clases_linknum (dict) diff_clases_linknum[i] is the name of the i-th category
	train_file='cfg/head_face_prioritytag_with_blur_data/train.txt'											
	diff_clases_linknum={ 
	0:'head',
	1:'face'
	}
	outputfile="../results_arpon/counting_classes_head_face_prioritytag_with_blur_data.txt"
	count_cat(train_file,diff_clases_linknum,outputfile)