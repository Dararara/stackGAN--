
import torch.nn as nn
def discriminator_loss(net_d, real_image, fake_image, sent_embs, real_labels, fake_labels):
    real_features = net_d(real_image)
    fake_features = net_d(fake_image)
    
    cond_real_logits = net_d.condition_DNET(real_features, sent_embs)
    #print('so far so good')
    #print(cond_real_logits, real_labels)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = net_d.condition_DNET(fake_features, sent_embs)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)

    batch_size = real_features.size(0)
    cond_wrong_logits = net_d.condition_DNET(real_features[:(batch_size - 1)], sent_embs[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    real_logits = net_d.uncondition_DNET(real_features)
    fake_logits = net_d.uncondition_DNET(fake_features)
    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    errD = ((real_errD + cond_real_errD) / 2.0 + 
            (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.0
            )
    return errD

def generator_loss(netDs, fake_images, sent_embs, real_labels):
    D_num = len(netDs)
    batch_size = real_labels.size(0)
    errG_total = 0
    for i in range(D_num):
        features = netDs[i](fake_images[i])
        condition_logits = netDs[i].condition_DNET(features, sent_embs)
        #print(condition_logits, real_labels)
        condition_errG = nn.BCELoss()(condition_logits, real_labels)
        logits = netDs[i].uncondition_DNET(features)
        
        
        errG = nn.BCELoss()(logits, real_labels)
        errG_total += errG
        
    return errG_total