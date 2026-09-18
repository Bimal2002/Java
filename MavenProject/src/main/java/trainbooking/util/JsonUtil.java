package trainbooking.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import trainbooking.model.User;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class JsonUtil {

    private static final ObjectMapper mapper = new ObjectMapper();

    public static List<User> readUsers(String filePath) throws Exception {

        File file = new File(filePath);

        if(!file.exists())
            return List.of();

        User[] users = mapper.readValue(file, User[].class);

        return Arrays.asList(users);
    }

    public static void writeUsers(String filePath, List<User> users) throws Exception {

        mapper.writerWithDefaultPrettyPrinter()
                .writeValue(new File(filePath), users);

    }

}