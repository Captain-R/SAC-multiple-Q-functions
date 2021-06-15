import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import matplotlib.pyplot as plt
import gym


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class Qnet(nn.Module):

    def __init__(self, inputsize, actionsize):
        super(Qnet, self).__init__()
        self.l1 = nn.Linear(inputsize + actionsize, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3.weight.data.uniform_(-3e-3, 3e-3)
        self.l3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        out = torch.cat([state, action], 1)
        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
        out = self.l3(out)

        return out


class Polnet(nn.Module):

    def __init__(self, inputsize, actionsize):
        super(Polnet, self).__init__()
        self.l1 = nn.Linear(inputsize, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, actionsize)
        self.logstd = nn.Linear(256, actionsize)

        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.logstd.weight.data.uniform_(-3e-3, 3e-3)
        self.logstd.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))

        mean = self.mean(out)
        logstd = self.logstd(out)
        logstd = torch.clamp(logstd, -20, 2)

        return mean, logstd

    def getactionentropy(self, state):
        mean, logstd = self.forward(state)
        n = Normal(mean, logstd.exp())
        s = n.rsample()  # rsample required
        action = torch.tanh(s)
        logpi = n.log_prob(s) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logpi


class Agent:

    def __init__(self, n, u, env, gamma, beta, tau, alpha, qlr, plr, alr):
        self.env = env
        self.n = n
        self.obsdim = env.observation_space.shape[0]
        self.actiondim = env.action_space.shape[0]
        self.gamma = gamma
        self.beta = beta
        self.updates = 0
        self.tempconsider = u
        self.tau = tau

        self.buffer = []

        self.qnets = []
        self.tqnets = []
        for i in range(self.n):
            self.qnets.append(Qnet(self.obsdim, self.actiondim).to(device))
            self.tqnets.append(Qnet(self.obsdim, self.actiondim).to(device))

        self.pnet = Polnet(self.obsdim, self.actiondim).to(device)

        for m in range(self.n):
            for i, j in zip(self.tqnets[m].parameters(), self.qnets[m].parameters()):
                i.data.copy_(j)

        self.qo = []

        for i in range(self.n):
            self.qo.append(torch.optim.Adam(self.qnets[i].parameters(), lr=qlr))

        self.po = torch.optim.Adam(self.pnet.parameters(), lr=plr)

        self.alpha = alpha
        if self.tempconsider == 1:
            self.logalpha = torch.zeros(1, requires_grad=True, device=device)
            self.ao = torch.optim.Adam([self.logalpha], lr=alr)
            self.targetentropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(device)).item()  # used usually

    def add(self, state, action, reward, nextstate, end):
        rewardarr = np.array([reward])
        data = (state, action, rewardarr, nextstate, end)
        self.buffer.append(data)

    def bufsample(self, batchsize):
        statelist = []
        actionlist = []
        nextstatelist = []
        rewardlist = []
        endlist = []

        samples = random.sample(self.buffer, batchsize)

        for p in samples:
            i, j, k, l, m = p
            statelist.append(i)
            actionlist.append(j)
            rewardlist.append(k)
            nextstatelist.append(l)
            endlist.append(m)

        sample = (statelist, actionlist, rewardlist, nextstatelist, endlist)
        return sample

    def action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, m = self.pnet.getactionentropy(state)
        action = action.cpu().detach().squeeze(0).numpy()
        p = [self.env.action_space.low, self.env.action_space.high]
        action = action * (p[1] - p[0]) / 2.0 + (p[1] + p[0]) / 2.0

        return action

    def update(self, batchsize):
        states, actions, rewards, nextstates, end = self.bufsample(batchsize)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        nextstates = torch.FloatTensor(nextstates).to(device)
        end = torch.FloatTensor(end).to(device)
        end = end.view(end.size(0), -1)

        newactions, logpi = self.pnet.getactionentropy(nextstates)

        newq = []
        for i in range(self.n):
            newq.append(self.tqnets[i].forward(nextstates, newactions))

        qnext = findmin(newq).to(device) - self.alpha * logpi
        qtarget = rewards + (1 - end) * self.gamma * qnext

        qloss = []
        for i in range(self.n):
            qloss.append(F.mse_loss(self.qnets[i].forward(states, actions), qtarget.detach()))

        for i in range(self.n):
            self.qo[i].zero_grad()
            qloss[i].backward()
            self.qo[i].step()

        polaction, logpi2 = self.pnet.getactionentropy(states)

        minq = []
        for i in range(self.n):
            minq.append(self.tqnets[i].forward(states, polaction))

        minestq = findmin(minq).to(device)
        ploss = (self.alpha * logpi2 - minestq).mean()

        self.po.zero_grad()
        ploss.backward()
        self.po.step()

        if self.tempconsider == 1:
            aloss = (self.logalpha * (-logpi2 - self.targetentropy).detach()).mean()

            self.ao.zero_grad()
            aloss.backward()
            self.ao.step()
            self.alpha = self.logalpha.exp()

        if self.updates % self.beta == 0:
            for m in range(self.n):
                for i, j in zip(self.tqnets[m].parameters(), self.qnets[m].parameters()):
                    i.data.copy_((self.tau * j) + ((1 - self.tau) * i))

        self.updates += 1


def act(agent, env, batchsize, maxeps, maxsteps, buf):
    eprewards = []
    eprewardslist = []
    for i in range(maxeps):
        print(i)
        truth = 0
        state = env.reset()
        episoderewardlist = []
        episodereward = 0
        for j in range(maxsteps):
            if j < buf:
                action = env.action_space.sample()
            else:
                action = agent.action(state)
                truth = 1

            nextstate, reward, end, _ = env.step(action)

            agent.add(state, action, reward, nextstate, end)

            if len(agent.buffer) > batchsize:
                agent.update(batchsize)

            if truth:
                episodereward += reward
                episoderewardlist.append(reward)
                truth = 0
            if end:
                break

            state = nextstate

        eprewards.append(episodereward)
        eprewardslist.append(episoderewardlist)
    return eprewards, eprewardslist


def process(lists):
    mini = []
    maxi = []
    mean = []
    
    for i in lists:
        mini.append(np.min(i))
        maxi.append(np.max(i))
        mean.append(np.mean(i))

    return mini, maxi, mean


def findmin(minq):
    minerq = minq[0]
    for i in range(len(minq) - 1):
        minerq = torch.min(minerq, minq[i + 1])
    return minerq


def plottotal(agent, numbers, values):
    for i in range(len(agent)):
        if i == 0:
            plt.plot(numbers, values[i], color=colors[i], label='Constant Temp.')
        if i == 1:
            plt.plot(numbers, values[i], color=colors[i], label='Variable Temp. with 2 Q func.')
        if i > 1:
            plt.plot(numbers, values[i], color=colors[i], label='Variable Temp. with ' + str(i + 1) + ' Q func.')

    plt.xlabel('Episode #')
    plt.ylabel('Total Reward')
    plt.title('Total rewards vs Episodes')
    plt.legend()
    plt.show()


def plotmax(agent, numbers, maxi):
    for i in range(len(agent)):
        if i == 0:
            plt.plot(numbers, maxi[i], color=colors[i], label='Constant Temp.')
        if i == 1:
            plt.plot(numbers, maxi[i], color=colors[i], label='Variable Temp. with 2 Q func.')
        if i > 1:
            plt.plot(numbers, maxi[i], color=colors[i], label='Variable Temp. with ' + str(i + 1) + ' Q func.')

    plt.xlabel('Episode #')
    plt.ylabel('Max reward of episode')
    plt.title('Max reward vs Episodes')
    plt.legend()
    plt.show()


def plotmin(agent, numbers, mini):
    for i in range(len(agent)):
        if i == 0:
            plt.plot(numbers, mini[i], color=colors[i], label='Constant Temp.')
        if i == 1:
            plt.plot(numbers, mini[i], color=colors[i], label='Variable Temp. with 2 Q func.')
        if i > 1:
            plt.plot(numbers, mini[i], color=colors[i], label='Variable Temp. with ' + str(i + 1) + ' Q func.')

    plt.xlabel('Episode #')
    plt.ylabel('Min reward of episode')
    plt.title('Min reward vs Episodes')
    plt.legend()
    plt.show()


def plotmean(agent, numbers, mean):
    for i in range(len(agent)):
        if i == 0:
            plt.plot(numbers, mean[i], color=colors[i], label='Constant Temp.')
        if i == 1:
            plt.plot(numbers, mean[i], color=colors[i], label='Variable Temp. with 2 Q func.')
        if i > 1:
            plt.plot(numbers, mean[i], color=colors[i], label='Variable Temp. with ' + str(i + 1) + ' Q func.')

    plt.xlabel('Episode #')
    plt.ylabel('Mean reward of episode')
    plt.title('Mean reward vs Episodes')
    plt.legend()
    plt.show()


def main():
    gamma = 0.99
    beta = 1
    alpha = 0.2
    qlr = 3e-4
    plr = 3e-4
    alr = 3e-4
    maxeps = 50
    maxsteps = 500
    batchsize = 60
    tau = 0.01
    k = 2
    buf = 50

    env = gym.make('MountainCarContinuous-v0')
    agent = []
    agent.append(Agent(2, 0, env, gamma, beta, tau, alpha, qlr, plr, alr))
    agent.append(Agent(2, 1, env, gamma, beta, tau, alpha, qlr, plr, alr))
    for n in range(k):
        agent.append(Agent(n + 3, 1, env, gamma, beta, tau, alpha, qlr, plr, alr))

    values = []
    lists = []
    for i in range(len(agent)):
        a, b = act(agent[i], env, batchsize, maxeps, maxsteps, buf)
        values.append(a)
        lists.append(b)

    mini = []
    maxi = []
    mean = []

    for i in range(len(lists)):
        a, b, c = process(lists[i])
        mini.append(a)
        maxi.append(b)
        mean.append(c)

    numbers = []
    for i in range(len(values[0])):
        numbers.append(i)

    plottotal(agent, numbers, values)
    plotmin(agent, numbers, mini)
    plotmax(agent, numbers, maxi)
    plotmean(agent, numbers, mean)
