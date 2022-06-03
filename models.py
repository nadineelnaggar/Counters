import torch
import torch.nn as nn


class LinearBracketCounter(nn.Module):
    def __init__(self, counter_input_size, counter_output_size, output_size, initialisation='random',output_activation='Sigmoid'):
        super(LinearBracketCounter, self).__init__()
        self.counter = nn.Linear(counter_input_size,counter_output_size, bias=False)
        self.out = nn.Linear(counter_output_size,output_size, bias=False)
        self.output_activation = output_activation
        if initialisation=='correct':
            self.counter.weight =  nn.Parameter(torch.tensor([[1, -1, 1]], dtype=torch.float32))
            self.out.weight = nn.Parameter(torch.tensor([[1]],dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()
        # self.clip = torch.clamp(min=0,max=1)

    def forward(self, x, previous_count):
        combined = torch.cat((x, previous_count))
        x = self.counter(combined)
        previous_count = x
        x = self.out(x)
        if self.output_activation=='Sigmoid':
            x = self.sigmoid(x)
        elif self.output_activation=='Clipping':
            x = torch.clamp(x,min=0,max=1)
        return x, previous_count


class NonZeroDyck1Counter(nn.Module):
    def __init__(self,counter_input_size, counter_output_size, output_size, initialisation='random',output_activation='Sigmoid'):
        super(NonZeroDyck1Counter, self).__init__()
        self.open_bracket_filter = nn.Linear(in_features=2,out_features=1,bias=False)
        # self.close_bracket_filter = nn.ReLU(nn.Linear(in_features=2,out_features=1,bias=False))
        self.close_bracket_filter = nn.Linear(in_features=2, out_features=1, bias=False)
        self.open_bracket_counter = nn.Linear(in_features=2,out_features=1,bias=False)
        self.close_bracket_counter = nn.Linear(in_features=2,out_features=1,bias=False)
        self.open_minus_close = nn.Linear(in_features=2,out_features=1,bias=False)
        self.close_minus_open = nn.Linear(in_features=2,out_features=1,bias=False)
        self.open_minus_close_copy = nn.Linear(in_features=1,out_features=1,bias=False)
        self.surplus_close_count = nn.Linear(in_features=2,out_features=1,bias=False)
        self.out = nn.Linear(in_features=2,out_features=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.output_activation=output_activation
        self.ReLU = nn.ReLU()
        if initialisation=='correct':
            self.open_bracket_filter.weight = nn.Parameter(torch.tensor([1,0],dtype=torch.float32))
            self.close_bracket_filter.weight = nn.Parameter(torch.tensor([0,1],dtype=torch.float32))
            self.open_bracket_counter.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
            self.close_bracket_counter.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
            self.open_minus_close.weight = nn.Parameter(torch.tensor([1,-1],dtype=torch.float32))
            self.close_minus_open.weight = nn.Parameter(torch.tensor([-1,1],dtype=torch.float32))
            self.open_minus_close_copy.weight = nn.Parameter(torch.tensor([1],dtype=torch.float32))
            self.surplus_close_count.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
            self.out.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))





    def forward(self,x, opening_brackets, closing_brackets, excess_closing_brackets):
        closing = self.close_bracket_filter(x)
        closing = self.ReLU(closing)
        # print(closing)
        # closing = self.closing_filter_relu(closing)

        closing = torch.cat((closing.unsqueeze(dim=0), closing_brackets.unsqueeze(dim=0)))
        closing = self.close_bracket_counter(closing)
        closing=self.ReLU(closing)
        # closing = self.closing_bracket_counter_relu(closing)
        closing_brackets = closing

        opening = self.open_bracket_filter(x)
        opening = self.ReLU(opening)
        # opening = self.opening_filter_relu(opening)

        opening = torch.cat((opening.unsqueeze(dim=0), opening_brackets.unsqueeze(dim=0)))
        opening = self.open_bracket_counter(opening)
        opening=self.ReLU(opening)
        # opening = self.open_bracket_counter_relu(opening)
        opening_brackets = opening

        # closing_minus_opening = torch.cat((closing.unsqueeze(dim=0), opening.unsqueeze(dim=0)))
        closing_minus_opening = torch.cat((opening.unsqueeze(dim=0), closing.unsqueeze(dim=0)))
        # opening_minus_closing = torch.cat((closing.unsqueeze(dim=0), opening.unsqueeze(dim=0)))
        opening_minus_closing = torch.cat((opening.unsqueeze(dim=0), closing.unsqueeze(dim=0)))
        closing_minus_opening = self.close_minus_open(closing_minus_opening)
        closing_minus_opening = self.ReLU(closing_minus_opening)
        # closing_minus_opening = self.closing_minus_opening_relu(closing_minus_opening)
        opening_minus_closing = self.open_minus_close(opening_minus_closing)
        opening_minus_closing=self.ReLU(opening_minus_closing)
        # opening_minus_closing = self.opening_minus_closing_relu(opening_minus_closing)

        opening_minus_closing = self.open_minus_close_copy(opening_minus_closing.unsqueeze(dim=0))
        opening_minus_closing=self.ReLU(opening_minus_closing)
        # opening_minus_closing = self.opening_minus_closing_copy_relu(opening_minus_closing)
        surplus_closing_brackets = torch.cat(
            (closing_minus_opening.unsqueeze(dim=0), excess_closing_brackets.unsqueeze(dim=0)))
        surplus_closing_brackets = self.surplus_close_count(surplus_closing_brackets)
        surplus_closing_brackets=self.ReLU(surplus_closing_brackets)
        # surplus_closing_brackets = self.closing_bracket_surplus_relu(surplus_closing_brackets)

        # output = torch.cat((surplus_closing_brackets.unsqueeze(dim=0), opening_minus_closing.unsqueeze(dim=0)))
        output = torch.cat((opening_minus_closing.unsqueeze(dim=0), surplus_closing_brackets.unsqueeze(dim=0)))
        output = self.out(output)
        # output = self.softmax(output)
        if self.output_activation=='Sigmoid':
            output=self.sigmoid(output)
        elif self.output_activation=='Clipping':
            output=torch.clamp(output,min=0,max=1)
        return output, opening_brackets, closing_brackets, surplus_closing_brackets


model = NonZeroDyck1Counter(counter_input_size=2, counter_output_size=1,output_size=1,initialisation='correct',output_activation='Clipping')

opening_brackets = torch.tensor(0,dtype=torch.float32)
closing_brackets = torch.tensor(0,dtype=torch.float32)
excess_closing_brackets = torch.tensor(0,dtype=torch.float32)
x = torch.tensor([1,0],dtype=torch.float32)

# print(model(x,opening_brackets,closing_brackets,excess_closing_brackets))
x = torch.tensor([[1,0],[0,1],[1,0],[1,0],[0,1],[0,1],[0,1],[1,0]],dtype=torch.float32)
for i in range(len(x)):
    print('x[i] = ',x[i])
    output, opening_brackets,closing_brackets, excess_closing_brackets = model(x[i],opening_brackets,closing_brackets,excess_closing_brackets)
    print('output = ',output)
    print('opening brackets = ',opening_brackets)
    print('closing brackets = ',closing_brackets)
    print('excess closing brackets = ',excess_closing_brackets)
    print('**********************')
