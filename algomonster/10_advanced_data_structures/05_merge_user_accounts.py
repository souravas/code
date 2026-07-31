class UnionFind:
    def __init__(self):
        self.id = {}

    def find(self, x):
        y = self.id.get(x, x)
        if y != x:
            self.id[x] = y = self.find(y)
        return y

    def union(self, x, y):
        self.id[self.find(x)] = self.find(y)


def merge_accounts(accounts: list[list[str]]) -> list[list[str]]:
    union_find = UnionFind()
    all_user_emails = set()
    for one_account in accounts:
        username = one_account[0]
        email_parent = None
        for email in one_account[1:]:
            user_email_pair = (username, email)
            all_user_emails.add(user_email_pair)
            if email_parent is None:
                email_parent = user_email_pair
            else:
                union_find.union(email_parent, user_email_pair)
    account_associations = {}
    for user_email_pair in all_user_emails:
        ancestor = union_find.find(user_email_pair)
        if ancestor not in account_associations:
            account_associations[ancestor] = []
        account_associations[ancestor].append(user_email_pair)
    return_res = []
    for user in account_associations:
        one_user = [user[0]]
        for email in sorted(account_associations[user]):
            one_user.append(email[1])
        return_res.append(one_user)
    return sorted(return_res, key=lambda a: (a[0], a[1]))
