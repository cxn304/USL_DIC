clc
clear
close all
 
n=6;  %Train dataset

SubsetSize = 256;
uu=zeros(SubsetSize,SubsetSize,2);
e=zeros(SubsetSize,SubsetSize,3);
shuliang=1;
for img = 1:60 

    xp=1:SubsetSize;
    yp=1:SubsetSize;
    [Xp_subset,Yp_subset] = meshgrid(xp,yp);

    % Define the regions size 
    for l = 1:n
        if l <2 % l==1
            s = 128;
        elseif  l<3 % l==2
            s = 64;
        elseif l<4 % l==3
            s = 32;
        elseif l<5
            s=16;
        elseif l<6
            s=8;
        else
            s=4;
        end
        name_image = sprintf('rf%04d.png',shuliang);
        Image_Ref = double(imread(name_image));
        xp0=[1:1/s:SubsetSize/s+1-1/s]+2;
        yp0=[1:1/s:SubsetSize/s+1-1/s]+2;
        xxp0=1:(SubsetSize/s)+3;
        yyp0=1:(SubsetSize/s)+3;
        [Xp_subset0,Yp_subset0] = meshgrid(xp0,yp0);

        % A random displamcent for each region 
        f =  randi([-100 100],SubsetSize/s+3,SubsetSize/s+3)/105;
        g =  randi([-100 100],SubsetSize/s+3,SubsetSize/s+3)/105;

        % Bicubic interpolation between the randoom displacments 
        x0 = Xp_subset0 ;
        y0 = Yp_subset0 ;
        disp_x = interp2(xxp0,yyp0,f,x0,y0,'bicubic');
        disp_y = interp2(xxp0,yyp0,g,x0,y0,'bicubic');

         % Setting the boundries dispalcements to 0
        disp_x(1:2,:) = 0;
        disp_y(1:2,:) = 0;        
        disp_x(:,1:2) = 0;
        disp_y(:,1:2) = 0;        
        disp_x(SubsetSize-1:SubsetSize,:) = 0;
        disp_y(SubsetSize-1:SubsetSize,:) = 0;        
        disp_x(:,SubsetSize-1:SubsetSize) = 0;
        disp_y(:,SubsetSize-1:SubsetSize) = 0;   
        uu(:,:,1)=disp_x;uu(:,:,2)=disp_y;
        e=calculate_strain(uu);
        uu=single(uu);e=single(e);
        % Generate the deformed image based on bi-cubic interpolation
        x = Xp_subset + disp_x;
        y = Yp_subset + disp_y;
        Image_BD = interp2(Xp_subset,Yp_subset,Image_Ref,x,y,'bicubic');
        Image_BD = Image_BD+4*randn(256,256);
        name_def = sprintf('df%04d.png',shuliang); 
        name_ue = sprintf('ue%04d.mat',shuliang);
        shuliang=shuliang+1;
        imwrite(uint8(Image_BD),name_def);
        save(name_ue,'uu','e');
     end 
end 

function e=calculate_strain(uu)
    [ex,uy]=gradient(uu(:,:,1));
    [vx,ey]=gradient(uu(:,:,2));
    exy = 0.5*(uy+vx);
    e=cat(3,ex,ey,exy);
end
%% 批量制作数据集
function train_data()
    imshow(cdata)
    [x,y]=ginput(1);
    x=int16(x);y=int16(y);
    n=0;
    for i = -3:3
        for j=-30:30
            n=n+1;
            name_rf = sprintf('rf%04d.png',n); 
            tmpimage=cdata(y+i:y+i+255,x+j:x+j+255);
            imwrite(tmpimage,name_rf);
        end
    end
end
