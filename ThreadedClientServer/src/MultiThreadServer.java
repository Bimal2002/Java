import java.io.*;
import java.net.*;

public class MultiThreadServer {

    public static void main(String[] args) {

        try {

            ServerSocket server = new ServerSocket(5000);
            System.out.println("Server started...");

            while (true) {

                Socket socket = server.accept();
                System.out.println("New client connected");

                ClientHandler client = new ClientHandler(socket);

                Thread thread = new Thread(client);
                thread.start();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}