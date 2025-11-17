import numpy as np

# Bloc de code de l'utilisateur (inchangé)
# -----------------------------------------------
a = np.array([1,2,3,4])
print(a)

a = np.array([[1,2],[3,4]])
print(a)

a = np.zeros(shape=(10))
print(a)
a = np.zeros(shape=(5,2))
print(a)

print(np.inf)

a = np.array([2,1,4,3])
print(np.max(a))
print(np.argmax(a))

l = [1,2,3,4]
print(l)
print(np.asarray(l))

# Array of Random integers ranging from 1 to 10 (with any size you want)
a = np.random.randint(low=1, high=10, size=(5,2))
print(a)

# Array of random elements of a list with any size you want
a = np.random.choice([0,1,2], size=(2,))

a = np.random.randint(low=1, high=5, size=(4,2))
print(a.shape)
print(a)

# Reshape a to a vector of shape = (8,1)
a = a.reshape((8,1))
print(a.shape)
print(a)


int_to_char = {
    0 : 'u',
    1 : 'r',
    2 : 'd',
    3 : 'l'
}

policy_one_step_look_ahead = {
    0 : [-1,0],
    1 : [0,1],
    2 : [1,0],
    3 : [0,-1]
}

def policy_int_to_char(pi,n):

    pi_char = ['']

    for i in range(n):
        for j in range(n):

            if i == 0 and j == 0 or i == n-1 and j == n-1:

                continue

            pi_char.append(int_to_char[pi[i,j]])

    pi_char.append('')

    return np.asarray(pi_char).reshape(n,n)


def policy_evaluation(n,pi,v,Gamma,threshhold):
  """
    This function should return the value function that follows the policy pi.
    Use the stopping criteria given in the problem statement.
    
    NOTE : 'Gamma' est ici le facteur d'actualisation (float).
  """
  gamma = Gamma 
  v_current = v.copy() 

  while True:
    delta = 0
    v_new = np.zeros((n, n)) 
    
    for i in range(n):
      for j in range(n):
        
        if (i == 0 and j == 0) or (i == n-1 and j == n-1):
          continue
          
        a = pi[i, j]
        
        next_i = i + policy_one_step_look_ahead[a][0]
        next_j = j + policy_one_step_look_ahead[a][1]
        
        reward = -1 
        
        if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
          next_i, next_j = i, j
          
        if (next_i == 0 and next_j == 0) or (next_i == n-1 and next_j == n-1):
          reward = 0
          
        v_new[i, j] = reward + gamma * v_current[next_i, next_j]
        
        delta = max(delta, abs(v_new[i, j] - v_current[i, j]))
        
    v_current = v_new 
    
    if delta < threshhold:
      break
      
  return v_current

def policy_improvement(n,pi,v,Gamma):
  """
    This function should return the new policy by acting in a greedy manner.
    The function should return as well a flag indicating if the output policy
    is the same as the input policy."""
    

  gamma = Gamma 
  new_pi = np.zeros((n, n), dtype=int)
  policy_stable = True
  
  for i in range(n):
    for j in range(n):
      
      if (i == 0 and j == 0) or (i == n-1 and j == n-1):
        continue
        
      old_action = pi[i, j]
      
      q_values = []
      for a in range(4): # 4 actions : 0, 1, 2, 3
        
        next_i = i + policy_one_step_look_ahead[a][0]
        next_j = j + policy_one_step_look_ahead[a][1]
        reward = -1
        
        if next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
          next_i, next_j = i, j
          
        if (next_i == 0 and next_j == 0) or (next_i == n-1 and next_j == n-1):
          reward = 0
          
        q_s_a = reward + gamma * v[next_i, next_j]
        q_values.append(q_s_a)
        
      best_action = np.argmax(q_values)
      new_pi[i, j] = best_action
      
      if old_action != best_action:
        policy_stable = False
        
  return new_pi, policy_stable

def policy_initialization(n):
  """
    This function should return the initial random policy for all states.
  """
  return np.zeros((n, n), dtype=int)

def policy_iteration(n,Gamma,threshhold):

    pi = policy_initialization(n=n)

    v = np.zeros(shape=(n,n))

    while True:

        v = policy_evaluation(n=n,v=v,pi=pi,threshhold=threshhold,Gamma=Gamma)

        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,Gamma=Gamma)

        if pi_stable:

            break

    return pi , v
    
    
n = 4

Gamma = [0.8,0.9,1]

threshhold = 1e-4

for _gamma in Gamma:

    pi , v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)

    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)

    print()

    print(pi_char)

    print()
    print()

    print(v)