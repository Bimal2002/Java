package trainbooking.service;

import trainbooking.model.User;
import trainbooking.util.JsonUtil;
import trainbooking.util.PasswordUtil;

import java.util.ArrayList;
import java.util.List;

public class UserService {

    private final String DB_PATH = "src/main/java/trainbooking/database/users.json";

    public void register(String username, String password) throws Exception {

        List<User> users = new ArrayList<>(JsonUtil.readUsers(DB_PATH));

        String hashed = PasswordUtil.hashPassword(password);

        users.add(new User(username, hashed));

        JsonUtil.writeUsers(DB_PATH, users);

        System.out.println("User Registered Successfully!");
    }

}