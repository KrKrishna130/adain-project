import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image

# is methods me args create kr skte hai jo cli se paas krni hai
# aage hm jitne n=bhi method hai agar unme koi bhi argument agar user pass krwani hai
# to yaha parse_arguments me ek args dfine krne honge
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default='/home/ubuntu/Desktop/NST_Code/content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default='/home/ubuntu/Desktop/NST_Code/style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default='/home/ubuntu/Desktop/NST_Code/vgg_normalised.pth',
                        help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')
    
    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop image')
    # batch size bhi user dalega
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    # lr --learning rate jb user dalga
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')
    

    return parser.parse_args()

# yaha hm main method de rahe hai,parse_arguments() bhi isme used krnge
def main():
    args = parse_arguments()
    # device koi bhi support k liye
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    #Save argument values,isse pta chlge kaun se experiment me kaun sa args liye the
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    #  get_transform - mtlb crop,short,reshape,resize krna image etc. hm kr rahe hai below method me
#    yaha hm 2 rakhe content,style k liye NST me leta hai ye
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    # ye dataset banae k liye use kr rahe hai--Dusre class wala call kiye hai
    #    yaha hm 2 rakhe content,style k liye NST me leta hai ye
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dateset = ImageFolderDataset(args.style_dir, style_transform)

# yaha ko content load kr rahe hai
    content_dataloader = DataLoader(content_dataset,
                                    batch_size=args.batch_size,
                                    shuffle = True,
                                    pin_memory=True,
                                    drop_last=True)
    # yaha ko style load kr rahe hai
    style_dataloader = DataLoader(style_dateset,
                                  batch_size=args.batch_size,
                                  shuffle=True, #mix krk fresh data deta hai train k liye
                                  pin_memory=True, # isse dataload me  hmesa GPU pr data pass krte hai,isse ye  process fast ho jata hai
                                  drop_last=True)  # drop last se jo value last me bache hai jinka batch size nii bn skta unko drop kr deta hai
    
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))
    
    # yaha hm model call krnge  encoder and decoder define krte hai
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

# yaha Optimizer lenge ,paramter ko kaise update krni hai ye decide krta hai
# lr=args.lr  args.lr hm upar me parse_arguments() method jo args diye the usko le rahe hai
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    # lr start me high hona chahiye bad me low hote jana chahiye
    # so shedular use krk dhire dhire time k sath low hote jaega lr
    # LambdaLR fun har epoch k bad lr chnage krega formula se
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch:  1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    print('Training...')

# now after training we calculated loss -MSELoss()

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None  # running loss
    running_closs = None  # running content loss
    running_sloss = None # running style loss

# tqdm is tracking the loop library

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))

        running_loss = 0
        running_closs = 0
        running_sloss = 0

       # it for one epoch data
        for content_batch, style_batch in progress_bar:
       # after one batch dataset comes then after then we move to GPU and perform operations
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

        # every images goes through encoder
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

       # here perform AdaIN Layer 2 layer--> c_feats,s_feats

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

    # this is O/P=G layer of AdaIN 
    # O/P is generated image
            g = decoder(t)

    # we take last layers for calculate loss ,for better training
            g_feats = encoder(g)

   # calculate content loss
            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
    # calculate style loss
            loss_s = loss_s * args.style_weight

    # total loss
            loss = loss_c + loss_s

# zero grad krk train hoga refresh value lega
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

        #schedular use to initial lr high after that lr decrease
        scheduler.step()

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)
# model save,check points,resume training points
# after some epochs interval time model will save
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

# yaha save ho raha hai,location=save_dir me
        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')  # decoder for save model after time interval
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')  # we load optimizer store value in dictionary
# we want some qualitive save 
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)




if __name__ == '__main__':
    main()