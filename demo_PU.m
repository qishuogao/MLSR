clc
close all
clear all


load PU
img=pavia_corrected;
gt=groundtruth;
classnum=9;
window_size=15;
patch_size=floor(window_size/2);

RandSampled_Num = 250*ones(1,9);


img_extend=padarray(img,[patch_size patch_size],'symmetric');  

scale_level=[0.1,0.2,0.3,0.4,0.5,0.7,1];
% scale_level=[0.1];

level_num=size(scale_level,2);
[I_row,I_line,I_high] = size(img);            
im = reshape(img,[I_row*I_line,I_high]);
im = im';

Indexall=[];
Labelall=[];

train_data=[];
train_label=[];
test_data=[];
test_label=[];
Index_train = [];
Index_test = [];
for i=1:classnum
    index=find(gt==i);
    label=ones(length(index),1)*i;
    Indexall=[Indexall ;index];
    Labelall=[Labelall; label];
    random_num=randperm(length(index));
    train_index = index(random_num(1:RandSampled_Num(i)));
    test_index=index(random_num(RandSampled_Num(i)+1:end));
    Index_train=[Index_train; train_index];
    Index_test=[Index_test; test_index];
    train_label=gt(Index_train);
    test_label=gt(Index_test);
    train_data=im(:,Index_train);
    test_data=im(:,Index_test);
    
end

train_data        =    train_data./ repmat(sqrt(sum(train_data.*train_data)),[size(train_data,1) 1]); 
test_data        =    test_data./ repmat(sqrt(sum(test_data.*test_data)),[size(test_data,1) 1]); 
im_data = im./ repmat(sqrt(sum(im.*im)),[size(im,1) 1]); 
xt=train_data';
[W]=w_compute(xt,train_label,0.3);


label=[];
for j=1:207400
     tt_data_multi={};
     
     [X,Y]=ind2sub(size(gt),j);
     X_new=X+patch_size;
     Y_new=Y+patch_size;
     X_range=[X_new-patch_size:X_new+patch_size];
     Y_range=[Y_new-patch_size:Y_new+patch_size];
     tt_data_neighbor=img_extend(X_range,Y_range,:);
     [tt_row,tt_col,tt_band]=size(tt_data_neighbor);
     tt_data_neighbor=reshape(tt_data_neighbor,tt_row*tt_col,tt_band);
     tt_data_neighbor=tt_data_neighbor';
     neighbor_num=tt_row*tt_col;
     tt_data_neighbor =  tt_data_neighbor./ repmat(sqrt(sum(tt_data_neighbor.*tt_data_neighbor)),[size(tt_data_neighbor,1) 1]);
     tt_data=im_data(:,j);
     
     x_test=tt_data';
     y_test=tt_data_neighbor';
     
     for k=1:level_num
         tt_temp_data=[];
         dis_matrix=[];
         num=[];
         for kk=1:neighbor_num
         [D]=d_compute(x_test,y_test(kk,:),W);
%          if D <=scale_level(k)
%             tt_temp_data=[tt_temp_data tt_data_neighbor(:,kk)];
%          end
         dis_matrix=[dis_matrix ;D];
         
         end
%          if tt_temp_data
         [dis_matrix_sort, I]=sort(dis_matrix);
         num(k)=floor(neighbor_num*scale_level(k));
         tt_data_multi{k}=tt_data_neighbor(:,I(1:num(k)));
     end  
        [label_test]=MLSR_function(train_data,tt_data_multi,3, level_num,train_label,classnum);
        label =  [label label_test];

      
    
end

map=reshape(label,610,340);
[acc] = ComputeClassificationAccuracy(map,gt)


