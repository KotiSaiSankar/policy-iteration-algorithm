# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.
## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.
## POLICY ITERATION ALGORITHM
Step1 :
We are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the MDP.

Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.

## POLICY IMPROVEMENT FUNCTION
### Name : Koti Sai Sankar
### Register Number : 212222240111
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
### Name : Koti Sai Sankar
### Register Number : 212222240111
```
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
![image](https://github.com/user-attachments/assets/870f54d4-3d0b-4f46-a38d-e492158654d4)
![image](https://github.com/user-attachments/assets/4e783823-115e-425c-9058-11c58882b005)
![image](https://github.com/user-attachments/assets/98503043-f9cc-4c2b-b98c-8c137de7c076)


### 2. Policy, Value function and success rate for the Improved Policy
![image](https://github.com/user-attachments/assets/eb8e4463-3885-4ac5-8106-e7d0708f3763)
![image](https://github.com/user-attachments/assets/90764b5e-ba48-45be-a166-6318e991e76a)
![image](https://github.com/user-attachments/assets/42ae03d9-39ef-448b-ba2f-e65349610c5c)


### 3. Policy, Value function and success rate after policy iteration
![image](https://github.com/user-attachments/assets/80cf1831-7372-478a-bee1-bafbb71ccf35)
![image](https://github.com/user-attachments/assets/83c65427-c0c5-4a71-bc68-5cd8f887e36b)
![image](https://github.com/user-attachments/assets/4d58b95d-741a-466f-b00d-2be95a46be7b)



## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
