parent(john, mary).
parent(mary, peter).

% หาปู่ของ peter
grandfather(X, peter) :-
  parent(X, mary).
