from torchvision import datasets, transforms
import torch
from dataset.aircraft import Aircraft
from dataset.cars196 import Cars
from dataset.cub200 import Cub2011

def get_dataloader(args):
    if args.dataset.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=True, download=False,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=False, download=False,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=8)
    elif args.dataset.lower()=='cifar100':
        if args.operator == 'PT':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_root, train=True, download=True,
                           transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_root, train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)
        elif args.operator == 'DF':
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_root, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)
    elif args.dataset.lower() in ['car196']:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_cars196 = Cars(args.data_root, train=True, transform=train_transform, download=True)
        test_car196 = Cars(args.data_root, train=False, transform=test_transform, download=True)
        print('length of train_cars196:', len(train_cars196))
        print('length of test_car196:', len(test_car196))
        if args.operator == 'PT':
            train_loader = torch.utils.data.DataLoader(
                dataset=train_cars196, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_car196, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8
            )
        elif args.operator == 'DF':
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                dataset=test_car196, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=4
            )

    elif args.dataset.lower() in ['cub200']:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_cub200 = Cub2011(args.data_root, train=True, transform=train_transform, download=True)
        test_cub200 = Cub2011(args.data_root, train=False, transform=test_transform, download=True)
        print('length of train_cub200:', len(train_cub200))
        print('length of test_cub200:', len(test_cub200))
        if args.operator == 'PT':
            train_loader = torch.utils.data.DataLoader(
                dataset=train_cub200, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_cub200, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8
            )
        elif args.operator == 'DF':
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                dataset=test_cub200, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8
            )
    elif args.dataset.lower() in ['aircraft']:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_aircraft = Aircraft(args.data_root, train=True, transform=train_transform, download=True)
        test_aircraft = Aircraft(args.data_root, train=False, transform=test_transform, download=True)
        print('length of train_aircraft:', len(train_aircraft))
        print('length of test_aircraft:', len(test_aircraft))
        if args.operator == 'PT':
            train_loader = torch.utils.data.DataLoader(
                dataset=train_aircraft, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_aircraft, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0
            )
        elif args.operator == 'DF':
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                dataset=test_aircraft, batch_size=args.test_batch_size, shuffle=True, pin_memory=False, num_workers=4
            )
    return train_loader, test_loader


if __name__ == '__main__':
    pass
