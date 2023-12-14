# Code adapted from https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole
import random
from random import randint
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
from joblib import Parallel, delayed
import itertools
from operator import add

class DoubleDQNAgent: # Define the agent that is going to be used for training
    def __init__(self, state_size, EPISODES,N, njam_max,Ma,Mm):
        # Define state and action space sizes
        self.state_size = state_size
        # Hyper-parameters for the Double-DQN architecture
        self.discount_factor = 0.99 # Discount factor for Bellman equation
        self.learning_rate = 0.001 # Learning rate for ADAM optimizer
        self.epsilon = 1.0 # Initial epsilon value (for epsilon greedy policy)
        self.epsilon_min = 0.01 # Minimal epsilon value (for epsilon greedy policy)
        self.epsilon_decay=(self.epsilon_min/self.epsilon)**(1/float(N*EPISODES*0.9))
        #self.epsilon_decay = 0.9999 # Epsilon decay (for epsilon greedy policy)
        #self.epsilon_decay=min(self.epsilon_decay,opt_decay)
        self.batch_size = 128 # Batch size for replay
        self.train_start = 250 # Adds a delay, for the memory to have data before starting the training
        # Create a replay memory using deque
        self.memory = deque(maxlen=10000)
        # CReate actions vectors
        self.actions_total=list() # Initialize list of actions
        # ACTIONS CODE: 0: jam, 1: njam, 2: attack, 3: no attack
        u_v_a=list()
        for i in range(pow(2,Ma)):
            bin_ch=bin(i)[2:]
            action_v=list(map(add,list(map(int, bin_ch.zfill(Ma))),[2]*Ma)) # To match the coding
            u_v_a.append(action_v)

        u_v_g=list()
        for i in range(pow(2,Mm)):
            bin_ch=bin(i)[2:]
            action_v=list(map(int, bin_ch.zfill(Mm)))
            u_v_g.append(action_v)

        for njam in range(njam_max+1):
            if njam==0:
                u_v_aux=list()
            else:
                u_v_aux=self.actions_total[njam-1].copy() #Use previous values and expand!
            for i in range(pow(2,Mm)): # For all good node actions
                if u_v_g[i].count(0)<=njam:
                    for j in range(pow(2,Ma)):
                        if u_v_aux.count(u_v_a[j]+u_v_g[i])==0: # Value NOT in list
                            u_v_aux.append(u_v_a[j]+u_v_g[i])
            self.actions_total.append(u_v_aux)
        self.action_size = len(self.actions_total[-1])
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model() # The target model is a NN used to increase stability
        # Initialize target model
        self.update_target_model()

    # NN input is the state, output is the estimated Q value for each action
    def build_model(self):
        # We build a model with 3 layers
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer='he_uniform'))
        #model.summary() # Uncomment to see the model summary provided by Keras
        # Compile the model: use Mean Squared Error as loss function, ADAM as optimizer
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Function to update the target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Epsilon greedy policy 
    def get_action(self, state):
        njam=state[0,-1]
        actions_actual=self.actions_total[njam]
        if np.random.rand() <= self.epsilon:
            return random.randrange(len(actions_actual))
        else:
            q_value = self.model.predict(state)
            q_value=q_value[0,0:len(actions_actual)]
            return np.argmax(q_value)

    # Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return # Start training only when there are some samples in the memory
        # Pick samples randomly from replay memory (with batch_size)
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        # Preprocess the batch by storing the data in different vectors
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0] #State
            action.append(mini_batch[i][1]) # Action
            reward.append(mini_batch[i][2]) # Reward
            update_target[i] = mini_batch[i][3] # Next state
            done.append(mini_batch[i][4]) # Done flag
        # Obtain the targets for the NN training phase
        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target) # Use the target network HERE for further stability

        for i in range(self.batch_size):
            # Get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                njam=int(update_target[i,-1])
                actions_actual=self.actions_total[njam]
                a = np.argmax(target_next[i,0:len(actions_actual)])
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])
        # Fit the model!
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
        
    def empirical_simulate(self,Mm,q,g,initial_state,Pc,u,N,Ma):
        state = np.reshape(initial_state, [1, self.state_size])
        rwd=0
        for stage in range(N):
            njam=state[0,-1]
            actions_actual=self.actions_total[njam]
            q_value = self.model.predict(state)
            q_value=q_value[0,0:len(actions_actual)]
            act_idx= np.argmax(q_value)  # returns only VALID actions!
            action_v=actions_actual[act_idx]
            next_state, reward = self.transition(Mm,q,g,state,action_v,Pc,u,N)
            rwd=rwd+reward/N
            next_state = np.reshape(next_state, [1, self.state_size])
            state = next_state  
        return rwd
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
    def transition(self,Mm,q,g,current_state,action_v,Pc,u,N):
        pe=Pc
            
        pe1=0
        pe2=0
        pe3=0
        _,state_size=current_state.shape
        state_size=int(state_size/2)
        rep_vector=np.arange(-N-1, N+2, 1)
        rep_g=current_state[0,0:state_size]
        rep_m=current_state[0,state_size:-1]
        njam=current_state[0,-1]
        Ma=sum(rep_m)
        actions_g=action_v[Ma:]
        actions_a=action_v[0:Ma]
        r_avg=sum(rep_g*rep_vector+rep_m*rep_vector)/sum(rep_g+rep_m)
        r=len(rep_vector); #Max reputation
        rep_stages=np.zeros(1, dtype=np.int)
        while (len(rep_stages)<=Mm+1):
            rg=rep_g[r-1]
            rm=rep_m[r-1]
            if rg+rm>0:
                for i in range(rg+rm):
                    rep_stages=np.append(rep_stages,rep_vector[r-1])
            r=r-1
            
        rep_stages=rep_stages[1:Mm+1] #Erase first value: was only for initialization
        
        #Make a decision
        dec=0
        W=0
        u_dec=None
        uiv=np.zeros(1, dtype=np.int)
        rep_g_aux=rep_g.copy() #To copy the list, copy() is required; otherwise, but using =, it is a reference to the original! (pointer)
        rep_m_aux=rep_m.copy()
        node=0 #Number of sensors called
        nodes_called=np.zeros(1, dtype=np.int) #Type of nodes called
        while (dec==0): #While decision is not taken
            #Call a sensor
            rep=rep_stages[node] #Reputation required
            node=node+1
            idr,=np.where(rep_vector==rep) # idr: reputation index
            rg=rep_g_aux[idr]
            rm=rep_m_aux[idr]
            node_idx_selected=randint(0,rg+rm-1)
            if node_idx_selected<rg: #Good node selected
                node_selected=1
                rep_g_aux[idr]=rep_g_aux[idr]-1 #Erase this sensor for future calls
            else:
                node_selected=-1
                rep_m_aux[idr]=rep_m_aux[idr]-1 #Erase this sensor for future calls
                
            if node==1: #Initialize nodes_called
                nodes_called=node_selected
            else:
                nodes_called=np.append(nodes_called,node_selected)
            
            # Error probability
            er=np.random.binomial(1,pe) #Node commits an error measuring
            # Generate decision by node
            if node_selected==1:  #Good node
                if actions_g[0]==1: # No jamming
                    if er==0:  # Good node, no error
                        ui=u
                    elif er==1: # Good node, error
                        ui=int(not(u))
                else: # There is jamming
                    ui=1
                    njam=njam-1
                    
                actions_g=actions_g[1:] #Erase action
            else: # Attacking sensor
                if actions_a[0]==3: #No attack
                    if er==0 : # no error
                        ui=u
                    elif er==1:  # error
                        ui=int(not(u))
                else: #Attack
                    if er==0:  #  no error
                        ui=int(not(u))
                    elif er==1: # error
                        ui=u
                        
                actions_a=actions_a[1:] #Erase action
        
            if node==1: #Initialize uiv
                uiv=ui
            else:
                uiv=np.append(uiv,ui)
            
            # Update W
            w=(rep+g)/(r_avg+g)
            if w<0:
                w=0
        
            W=W+pow(-1,float(ui)+1)*w;
            # Decide
            if W>=q:
                u_dec=1
                dec=1
                pe1=1
            elif W<=-q:
                u_dec=0
                dec=1
                pe2=1
            elif node==Mm:
                u_dec=1
                dec=2
                pe3=1
        
        #Update reputations
        if type(uiv)==int: #Only one sensor has been called!!
            if u_dec==uiv: # Reputation increase
                rep=rep_stages[0]+1 # Reputation increases by one unit!
                idr,=np.where(rep_vector==rep) # idr: reputation index
                if nodes_called==1: #Good node
                    rep_g_aux[idr]=rep_g_aux[idr]+1
                else:
                    rep_m_aux[idr]=rep_m_aux[idr]+1
            else: #Repuation decrease!
                rep=rep_stages[0]-1 # Reputation decreases by one unit!
                idr,=np.where(rep_vector==rep) # idr: reputation index
                if nodes_called==1: #Good node
                    rep_g_aux[idr]=rep_g_aux[idr]+1
                else:
                    rep_m_aux[idr]=rep_m_aux[idr]+1
        else:
            for i in range(len(uiv)):
                if u_dec==uiv[i]: # Reputation increase
                    rep=rep_stages[i]+1 # Reputation increases by one unit!
                    idr,=np.where(rep_vector==rep) # idr: reputation index
                    if nodes_called[i]==1: #Good node
                        rep_g_aux[idr]=rep_g_aux[idr]+1
                    else:
                        rep_m_aux[idr]=rep_m_aux[idr]+1
                else: #Repuation decrease!
                    rep=rep_stages[i]-1 # Reputation decreases by one unit!
                    idr,=np.where(rep_vector==rep) # idr: reputation index
                    if nodes_called[i]==1: #Good node
                        rep_g_aux[idr]=rep_g_aux[idr]+1
                    else:
                        rep_m_aux[idr]=rep_m_aux[idr]+1
        # Output values
        if u==0:
            reward=pe1+pe3
        else:
            reward=pe2
        
        next_state=np.append(rep_g_aux, rep_m_aux)
        next_state=np.append(next_state, njam)
        
        return (next_state,reward)

if __name__ == "__main__":
    def processinput(Pc,u):
        nj=0
        nsm=dic[str(nj)+" "+str(Pc)+" "+str(u)]
        if os.path.isfile("./python_data/training"+str(nsm)+".h5")==True: #NN trained already
                        print("Simulation: {}/{} already stored"
                                          .format(nsm,nts))
        else:
            N=5 # Number of iterations
            Ma=1
            M=10
            Mg=M-Ma
            njam_max=nj
            # EWSZOT parameters
            Mm=4
            q=2
            g=5.51
            # Network parameters
            EPISODES = 2000 
            
            state_size =2*(2*(N+1)+1)+1
            initial_state=np.zeros(state_size, dtype=np.int)
            initial_state[N+1]=Mg
            sz=int((state_size-1)/2)
            initial_state[sz+N+1]=Ma
            initial_state[-1]=njam_max
            agent = DoubleDQNAgent(state_size, EPISODES,N, njam_max,Ma,Mm)
            scores, episodes, q_trained = [], [], [] # To store values for plotting
        
            for e in range(EPISODES):
                
                done = False
                score = 0
                state = initial_state
                state = np.reshape(state, [1, state_size])
                stage=0
                while not done: # Iterate while the game has not finished
                    stage=stage+1
                    # Get action for the current state and go one step in environment
                    action = agent.get_action(state) # Using epsilon-greedy policy
                    nj_id=state[0,-1] # Use only possible actions!!
                    actions_actual=agent.actions_total[nj_id]
                    action_v=actions_actual[action]
                    next_state, reward = agent.transition(Mm,q,g,state,action_v,Pc,u,N)
                    reward=reward/N # Reward averaged through stages
                    if stage==N:
                        done=True #Final state!!
                    next_state = np.reshape(next_state, [1, state_size])
            
                    # Save the sample <s, a, r, s'> to the replay memory
                    agent.append_sample(state, action, reward, next_state, done)
                    # Train
                    agent.train_model()
                    score += reward
                    state = next_state
                    if done:
                        # Update target model after each episode
                        agent.update_target_model()
                        # Store values for plotting
                        scores.append(score)
                        episodes.append(e)
                        state = initial_state
                        state = np.reshape(state, [1, state_size])
                        q_trained.append(np.amax(agent.model.predict(state)))
                        # Output the results of the episode
                        if e % 250 ==0:
                            print("episode:", e, "  mean score:", np.mean(scores[-min(20, len(scores)):]), "  memory length:",
                                  len(agent.memory), "  epsilon:", agent.epsilon)
        
        
            # Plot evolution
            #plt.plot(episodes, scores)
            #plt.show()
            #plt.plot(episodes, q_trained)
            #plt.show()
            #Empirical check of values
            nsim=100
            reward_emp=0
            rwdv=np.zeros(nsim)
            for sim in range(nsim):
                rwd=agent.empirical_simulate(Mm,q,g,initial_state,Pc,u,N,Ma)
                reward_emp=reward_emp+rwd/nsim
                rwdv[sim]=rwd
                
            print("Empirical reward obtained: {}"
                              .format(reward_emp))
            # Save trained model and Matlab data
            agent.save("./python_data/training"+str(nsm)+".h5")
            out_values={} #Empty dictionary
            out_values["M"]=M
            out_values["Ma"]=Ma
            out_values["Mg"]=Mg
            out_values["Pc"]=Pc
            out_values["u"]=u
            out_values["Mm"]=Mm
            out_values["N"]=N
            out_values["q"]=q
            out_values["g"]=g
            out_values["nj"]=nj
            out_values["reward_emp"]=reward_emp
            sio.savemat("./python_data/DQN_"+str(nsm)+".mat",out_values)
            print("Simulation: {}/{} finished"
                              .format(nsm,nts))
    Pcv=list(np.linspace(0,0.5,51))
    uv=[0,1]
    njv=[0,1,2]
    global nts
    nts=len(Pcv)*len(uv)*len(njv)
    # Create dictinoary to save data!!
    global dic
    dic={}
    nsm=0
    for nj,Pc,u in itertools.product(njv,Pcv,uv):
        key=str(nj)+" "+str(Pc)+" "+str(u)
        nsm=nsm+1
        dic[key]=nsm
    # Parallel for to train!
    num_cores = 1 # To use all cores
    results = Parallel(n_jobs=num_cores, verbose=5)(delayed(processinput)(Pc=k,u=l) for k in Pcv for l in uv)
