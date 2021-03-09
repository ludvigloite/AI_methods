import numpy as np

def forward(f0,t,O,T,E):

    f = f0

    for k in range(t):
        if E[k] is not None:
            f = np.diag(O[E[k]]) @ T.T @ f
        else:
            f = T.T @ f
        t += 1

    return f


def backward(N,t,O,T,E):

    b = np.ones(T[0].shape)

    for i in range(N-1,t-1,-1):
        e = E[i]
        O_diag = np.diag(O[e])
        b = T @ O_diag @ b

    return b


def forward_backward(N,f0,t,O,T,E):

    fv = forward(f0,t,O,T,E)
    b = backward(N,t,O,T,E)
    sv = fv * b
    sv = sv / sv.sum()

    return sv


def viterbi(f0,O,T,E):
    H, _ = T.shape
    N = len(E)
    M = np.zeros((H, N))

    M0 = O[E[0]] * (T.T @ f0)

    M[:,0] = M0 / M0.sum() #Not really sure why we need to normalize this, and not the other elements
    

    for i in range(1,N):
        M[:,i] = O[E[0]] * np.max(T * M[:,None,i-1], axis=0)

    bestPath = np.argmax(M, axis = 0)
    return bestPath, M


def forwardDBN(f0,t,Track,Food,T,E):

    f = f0

    for k in range(t):
        #print("E: ",E[k])
        if E[k,0] is not None:
            f = np.diag(Track[E[k,0]]) @ np.diag(Food[E[k,1]]) @ T.T @ f
        else:
            f = T.T @ f
        t += 1

    return f


def backwardDBN(N,t,Track,Food,T,E):

    b = np.ones(2)

    for i in range(N-1,t-1,-1):
        Track_diag = np.diag(Track[E[i,0]])
        Food_diag = np.diag(Food[E[i,1]])
        b = T @ Track_diag @ Food_diag @ b

    return b


def forward_backwardDBN(N,f0,t,Track,Food,T,E):

    fv = forwardDBN(f0,t,Track,Food,T,E)
    b = backwardDBN(N,t,Track,Food,T,E)
    sv = fv * b
    sv = sv / sv.sum()

    return sv


def task1():

    print("\n\n--- TASK 1 ---")

    runB = True
    runC = True
    runD = True
    runE = True

    #Emission table
    O = np.array([[0.75, 0.2],
                    [0.25, 0.8]])

    # Transition table
    T = np.array([[0.8, 0.3],
                    [0.2, 0.7]]).T

    # Initial estimate
    f0 = np.array([0.5, 0.5])

    # Evidence. Figured i need: Birds = 0, no birds = 1
    E = np.array([0,0,1,0,1,0])

    if runB:

        print("\n\n--- Task 1b - Filtering ---")

        for t in range(6):
            t += 1

            f = forward(f0,t,O,T,E)
            f = f/f.sum()

            #print("\nt = ",t)
            print(f"P(X{t} | e1:{t}) = {f}")
        
    if runC:

        print("\n\n--- Task 1c - Prediction ---")

        E_pred = [0, 0, 1, 0, 1, 0] + [None]*24

        for t in range(6,30):
            t += 1

            f = forward(f0,t,O,T,E_pred)
            f = f/f.sum()

            #print("\nt = ",t)
            print(f"P(X{t} | e1:6) = {f}")

    if runD:

        print("\n\n--- Task 1d - Smoothing ---")

        N = 6


        for t in range(N):
            sv = forward_backward(N,f0,t,O,T,E)

            #print("\nt = ",t)
            print(f"P(X{t} | e1:{N}) = {sv}")


    if runE:

        print("\n\n--- Task 1e - Most likely sequence ---")

        bestPath, M = viterbi(f0,O,T,E)
        bestPathBool = []

        for i in range(len(bestPath)):
            if bestPath[i]==0:
                bestPathBool.append(True)
            else:
                bestPathBool.append(False)

        print(f"\nThe most likely sequence is:\n {bestPathBool}\n")
        print(f"The probabilities (M matrix) are: \n {M}\n")


def task2():

    print("\n\n--- TASK 2 ---")

    runB = True
    runC = True
    runE = True

    #Emission tables
    Track = np.array([[0.7, 0.2],
                    [0.3, 0.8]])

    Food = np.array([[0.3, 0.1],
                    [0.7, 0.9]])


    # Transition table
    Trans = np.array([[0.8, 0.3],
                    [0.2, 0.7]]).T

    # Initial estimate
    f0 = np.array([0.7, 0.3])

    # Evidence. Figured i need: True = 0, False = 1
    E = np.array([[0, 0],
                [1, 0],
                [1, 1],
                [0, 1]])


    #print(np.diag(Track[E[2,0]]))

    if runB:

        print("\n\n--- Task 2b - Filtering ---")

        for t in range(4):
            t += 1

            f = forwardDBN(f0,t,Track,Food,Trans,E)
            f = f/f.sum()

            print("\nt = ",t)
            print(f"P(X{t} | e1:{t}) = {f}")


    if runC:

        print("\n\n--- Task 2c - Prediction ---")

        E_pred = np.array([[0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [None, None],
                [None, None],
                [None, None],
                [None, None]])

        for t in range(4,8):
            t += 1

            f = forwardDBN(f0,t,Track,Food,Trans,E_pred)
            f = f/f.sum()

            print("\nt = ",t)
            print(f"P(X{t} | e1:4) = {f}")

    if runE:

        print("\n\n--- Task 2e - Smoothing ---")

        N = 4

        for t in range(N):
            sv = forward_backwardDBN(N,f0,t,Track,Food,Trans,E)

            print("\nt = ",t)
            print(f"P(X{t} | e1:{N}) = {sv}")


    


    




if __name__ == "__main__":
    ##

    runTask1 = False
    runTask2 = True

    if runTask1:
        task1()

    if runTask2:
        task2()

    


