import java.io.*;
import java.net.*;

public class ClientHandler implements Runnable {

    private Socket socket;

    public ClientHandler(Socket socket) {
        this.socket = socket;
    }

    public void run() {

        try {

            BufferedReader in = new BufferedReader(
                    new InputStreamReader(socket.getInputStream()));

            PrintWriter out = new PrintWriter(
                    socket.getOutputStream(), true);

            String msg = in.readLine();

            System.out.println("Client says: " + msg);

            out.println("Message received");

            socket.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}