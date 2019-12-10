#functions:将含有gatelayer层剪枝后的网络，根据gatelayer里面的0，1进行前后网络的裁剪；
#result:生成剪枝后的网络，但是gatelayer层仍然保留，只是0的部分裁剪了，后面需要根据权重，生成对应的网络结构；

import torch
import argparse
from models.resnet_18 import resnet18
from models.resnet import resnet101

def load_model_pytorch(model, load_model, model_name):
    print("=> loading checkpoint '{}'".format(load_model))
    checkpoint = torch.load(load_model)

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    # match_dictionaries, useful if loading model without gate:
    if 'module.' in list(model.state_dict().keys())[0]:
        if 'module.' not in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

    if 'module.' not in list(model.state_dict().keys())[0]:
        if 'module.' in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    # just for vgg
    if model_name == "vgg":
        from collections import OrderedDict

        load_from = OrderedDict([(k.replace("features.", "features"), v) for k, v in load_from.items()])
        load_from = OrderedDict([(k.replace("classifier.", "classifier"), v) for k, v in load_from.items()])

    if 1:
        for ind, (key, item) in enumerate(model.state_dict().items()):
            if ind > 10:
                continue
            print(key, model.state_dict()[key].shape)

        print("*********")

        for ind, (key, item) in enumerate(load_from.items()):
            if ind > 10:
                continue
            print(key, load_from[key].shape)

    for key, item in model.state_dict().items():
        # if we add gate that is not in the saved file
        if key not in load_from:
            load_from[key] = item
        # if load pretrined model
        if load_from[key].shape != item.shape:
            load_from[key] = item

    model.load_state_dict(load_from, strict=True)
    
    epoch_from = -1
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(load_model, epoch_from))

def dynamic_network_change_local(model):
    '''
    Methods attempts to modify network in place by removing pruned filters.
    Works with ResNet101 for now only
    :param model: reference to torch model to be modified
    :return:
    '''
    # change network dynamically given a pruning mask

    # step 1: model adjustment
    # lets go layer by layer and get the mask if we have parameter in pruning settings:

    pruning_maks_input = None
    prev_model1 = None
    prev_model2 = None
    prev_model3 = None

    pruning_mask_indexes = None
    gate_track = -1

    skip_connections = list()

    current_skip = 0
    DO_SKIP = True

    gate_size = -1
    current_skip_mask_size = -1

    for module_indx, m in enumerate(model.modules()):
        pruning_mask_indexes = None
        if not hasattr(m, "do_not_update"):
            if isinstance(m, torch.nn.Conv2d):
                if 1:
                    if pruning_maks_input is not None:
                        print("fixing interm layer", gate_track, module_indx)

                        m.weight.data = m.weight.data[:, pruning_maks_input]

                        pruning_maks_input = None
                        print("weight size now", m.weight.data.shape)
                        m.in_channels = m.weight.data.shape[1]
                        m.out_channels = m.weight.data.shape[0]

                        if DO_SKIP:
                            print("doing skip connection")
                            if m.weight.data.shape[0] == current_skip_mask_size:
                                m.weight.data = m.weight.data[current_skip_mask]
                                print("weight size after skip", m.weight.data.shape)

                    if module_indx in [23, 63, 115, 395]:
                        if DO_SKIP:
                            print("fixing interm skip")
                            m.weight.data = m.weight.data[:, prev_skip_mask]

                            print("weight size now", m.weight.data.shape)
                            m.in_channels = m.weight.data.shape[1]
                            m.out_channels = m.weight.data.shape[0]

                            if DO_SKIP:
                                print("doing skip connection")
                                if m.weight.data.shape[0] == current_skip_mask_size:
                                    m.weight.data = m.weight.data[current_skip_mask]
                                    print("weight size after skip", m.weight.data.shape)

            if isinstance(m, torch.nn.BatchNorm2d):
                print("interm layer BN: ", gate_track, module_indx)
                if DO_SKIP:
                    print("doing skip connection")
                    if m.weight.data.shape[0] == current_skip_mask_size:
                        m.weight.data = m.weight.data[current_skip_mask]
                        print("weight size after skip", m.weight.data.shape)

                        # m.weight.data = m.weight.data[current_skip_mask]
                        m.bias.data = m.bias.data[current_skip_mask]
                        m.running_mean.data = m.running_mean.data[current_skip_mask]
                        m.running_var.data = m.running_var.data[current_skip_mask]
        else:
            # keeping track of gates:
            gate_track += 1

            then_pass = False
            if gate_track < 4:
                # skipping skip connections
                then_pass = True
                skip_connections.append(m.weight)
                current_skip = -1

            if not then_pass:
                pruning_mask = m.weight
                if gate_size!=m.weight.shape[0]:
                    current_skip += 1
                    current_skip_mask_size = skip_connections[current_skip].data.shape[0]

                    if skip_connections[current_skip].data.shape[0] != 2048:
                        current_skip_mask = skip_connections[current_skip].data.nonzero().view(-1)
                    else:
                        current_skip_mask = (skip_connections[current_skip].data + 1.0).nonzero().view(-1)
                    prev_skip_mask_size = 64
                    prev_skip_mask = range(64)
                    if current_skip > 0:
                        prev_skip_mask_size = skip_connections[current_skip - 1].data.shape[0]
                        prev_skip_mask = skip_connections[current_skip - 1].data.nonzero().view(-1)

                gate_size = m.weight.shape[0]

                if 1:
                    print("fixing layer", gate_track, module_indx)
                    if 1.0 in pruning_mask:
                        pruning_mask_indexes = pruning_mask.nonzero().view(-1)
                    else:
                        pruning_mask_indexes = []
                    m.weight.data = m.weight.data[pruning_mask_indexes]
                    for prev_model in [prev_model1, prev_model2, prev_model3]:
                        if isinstance(prev_model, torch.nn.Conv2d):
                            print("prev fixing layer", prev_model, gate_track, module_indx)
                            prev_model.weight.data = prev_model.weight.data[pruning_mask_indexes]
                            print("weight size", prev_model.weight.data.shape)

                            if DO_SKIP:
                                print("doing skip connection")

                                if prev_model.weight.data.shape[1] == current_skip_mask_size:
                                    prev_model.weight.data = prev_model.weight.data[:, current_skip_mask]
                                    print("weight size", prev_model.weight.data.shape)

                                if module_indx in [53, 105, 385]:  # add one more layer for this transition
                                    print("doing skip connection")

                                    if prev_model.weight.data.shape[1] == prev_skip_mask_size:
                                        prev_model.weight.data = prev_model.weight.data[:, prev_skip_mask]
                                        print("weight size", prev_model.weight.data.shape)

                        if isinstance(prev_model, torch.nn.BatchNorm2d):
                            print("prev fixing layer", prev_model, gate_track, module_indx)
                            prev_model.weight.data = prev_model.weight.data[pruning_mask_indexes]
                            prev_model.bias.data = prev_model.bias.data[pruning_mask_indexes]
                            prev_model.running_mean.data = prev_model.running_mean.data[pruning_mask_indexes]
                            prev_model.running_var.data = prev_model.running_var.data[pruning_mask_indexes]

                pruning_maks_input = pruning_mask_indexes

        prev_model3 = prev_model2
        prev_model2 = prev_model1
        prev_model1 = m

    if DO_SKIP:
        # fix gate layers
        gate_track = 0

        for module_indx, m in enumerate(model.modules()):
            if hasattr(m, "do_not_update"):
                gate_track += 1
                if gate_track < 4:
                    if m.weight.shape[0] < 2048:
                        m.weight.data = m.weight.data[m.weight.nonzero().view(-1)]

    print("printing conv layers")
    for module_indx, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.Conv2d):
            print(module_indx, "->", m.weight.data.shape)

    print("printing bn layers")
    for module_indx, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            print(module_indx, "->", m.weight.data.shape)

    print("printing gate layers")
    for module_indx, m in enumerate(model.modules()):
        if hasattr(m, "do_not_update"):
            print(module_indx, "->", m.weight.data.shape, m.size_mask)

def make_model(model,model_path):
    #read_weight
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict,False)
    model = model.cuda()
    return model

def save_model(model,arch,save_weight_path):
    torch.save({'arch': arch,
                'state_dict': model.state_dict()},save_weight_path)

def main():
    parser = argparse.ArgumentParser(description='Pruned process gatelayer')
    parser.add_argument('--load_weights_path', default='', type=str,
                        help='path to pruned model weights')
    parser.add_argument('--model_arch', default='', type=str,
                        help='name of model_arch')
    parser.add_argument('--save_weight_path', default='', type=str,
                        help='path to save weights')

    args = parser.parse_args()
    if args.model_arch == 'resnet18':
        model = resnet18(num_classes=1000)
    elif args.model_arch == 'resnet101':
        model = resnet101(num_classes=1000)

    model = make_model(model,args.load_weights_path)
    dynamic_network_change_local(model)
    save_model(model,args.model_arch,args.save_weight_path)

if __name__ == '__main__':
    
    main()