from board import Board
from markov_decision_process.policy import Policy
from markov_decision_process.db import DB

omega = Board()
db = DB(db_name='omega', top_exceed=1000000)
db.load_data()
policy = Policy(gamma=0.9, tao=0.1, db=db,
                default_value=0)
steps = 10000000
for i in range(steps):
    print i
    if (i+1) % 100 == 0:
        db.bulk_save()
    while omega.ended:
        actions = omega.possibleSteps
        if omega.step < 52:
            ended = False
        else:
            winner, ended = omega.judge_terminal()
        # print '\n train huhihutu called'
        # print ('actions:', actions)
        # print('winner: ',winner,' ended: ',ended)
        if actions.shape[0] == 0 and ended == False:
            omega.play()
            continue
        else:
            if omega.color == -1:
                (action, value, move) = omega.apply_policy(policy.pai, 'max')
            else:
                (action, value, move) = omega.apply_policy(policy.pai, 'min')
            if value != 0.0:
                db.store_state(omega.board, value)
            omega.play(action)
    omega.reset()
