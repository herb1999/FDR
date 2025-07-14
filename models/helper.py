from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import logging


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    # tqdm_gen = tqdm(trainloader)
    # for i, batch in enumerate(tqdm_gen, 1):
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        train_label = train_label.long()  # 确保标签是 LongTensor

        logits = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def extract_embeddings(trainset, transform, model, args):
    model = model.eval()
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False
    )
    trainloader.dataset.transform = transform
    embeddings, labels = [], []

    with torch.no_grad():
        for data, label in tqdm(trainloader, desc="Extracting embeddings"):
            data = data.cuda()
            model.module.mode = 'encoder'
            embedding = model(data).cpu()
            embeddings.append(embedding)
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels


def build_base_prototypes(embeddings, labels, base_classes, split_idx):
    prototypes = []
    for class_idx in range(split_idx):
        mask = (labels == class_idx)
        class_embeddings = embeddings[mask]
        prototypes.append(class_embeddings.mean(dim=0))
    return torch.stack(prototypes, dim=0)


def build_novel_prototypes(embeddings, labels, base_classes, split_idx, n_shot, epoch):
    prototypes = []
    # 添加epoch相关的随机种子
    g = torch.Generator()
    g.manual_seed(torch.initial_seed() + epoch * 1000)  # 每个epoch不同
    for class_idx in range(split_idx, base_classes):
        mask = (labels == class_idx)
        class_embeddings = embeddings[mask]

        if len(class_embeddings) == 0:
            proto = torch.zeros_like(embeddings[0])
        else:
            perm = torch.randperm(len(class_embeddings), generator=g)
            selected = class_embeddings[perm[:n_shot]]
            proto = selected.mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes, dim=0)


def replace_base_fc_split(trainset, transform, model, args):
    embeddings, labels = extract_embeddings(trainset, transform, model, args)

    # 保存后半部分类别的GT原型（全样本均值）
    gt_novel_prototypes = []
    for class_idx in range(args.split_idx, args.base_class):
        mask = (labels == class_idx)
        class_embeddings = embeddings[mask]
        gt_novel_prototypes.append(class_embeddings.mean(dim=0))
    model.module.gt_novel_prototypes = torch.stack(gt_novel_prototypes, dim=0).cuda()

    # 更新基类原型（前半部分）
    base_prototypes = build_base_prototypes(embeddings, labels, args.base_class, args.split_idx)
    model.module.fc.weight.data[:args.split_idx] = base_prototypes

    # 初始新类原型（可选，或在每个epoch更新）
    novel_prototypes = build_novel_prototypes(embeddings, labels, args.base_class, args.split_idx, args.shot, epoch=0)
    model.module.fc.weight.data[args.split_idx:args.base_class] = novel_prototypes

    return model,embeddings, labels


# 每个epoch调用此函数更新新类原型
def update_novel_prototypes(model, embeddings, labels, args, epoch):
    split_idx = args.split_idx
    novel_prototypes = build_novel_prototypes(embeddings, labels, args.base_class, split_idx, args.shot, epoch)
    model.module.fc.weight.data[split_idx:args.base_class] = novel_prototypes
    return model


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(trainloader, desc="get embedding")):
            data, label = batch
            data = data.cuda()
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label)

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, result_list=None):
    print('Testing session {}'.format(session))
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader), 1):
            data, test_label = batch
            data,test_label = data.cuda(),test_label.cuda()
            # data, test_label = [_.cuda() for _ in batch]
            # test_label = test_label.long()  # 确保标签是 LongTensor
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()

        logging.info('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)

        # if session > 0:
        #     _preds = torch.argmax(lgt, dim=1)
        #     torch.save(_preds, f"pred_labels/{args.project}_{args.dataset}_{session}_preds.pt")
        #     torch.save(lbs, f"pred_labels/{args.project}_{args.dataset}_{session}_labels.pt")
        #     torch.save(model.module.fc.weight.data.cpu()[:test_class], f"pred_labels/{args.project}_{args.dataset}_{session}_weights.pt")

        if session > 0:
            # save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs)
            perclassacc = cm.diagonal()
            # print(perclassacc)
            # print(np.mean(perclassacc))
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])

            result_list.append(f"Seen Acc:{seenac}  Unseen Acc:{unseenac}")
            return vl, (seenac, unseenac, va)
        else:
            return vl, va