package repository;

import models.User;
import utils.JsonUtils;

import java.util.List;
import java.util.Locale;
import java.util.Optional;

public class UserRepository {
    public Optional<User> findByEmail(String email) {
        return JsonUtils.readUsers()
                .stream()
                .filter(user -> user.getEmail() != null && user.getEmail().toLowerCase(Locale.ROOT).equals(email.toLowerCase(Locale.ROOT)))
                .findFirst();
    }

    public Optional<User> findById(String id) {
        return JsonUtils.readUsers()
                .stream()
                .filter(user -> user.getId() != null && user.getId().equals(id))
                .findFirst();
    }

    public void save(User user) {
        JsonUtils.appendUser(user);
    }

    public List<User> findAll() {
        return JsonUtils.readUsers();
    }
}
