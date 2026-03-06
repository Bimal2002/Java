package db;
import java.sql.Connection;
import java.sql.DriverManager;

public class DBConnection {
    private static final String URL = "jdbc:sqlite:train_booking.db";

    public static Connection connect(){
        try{
            Connection conn = DriverManager.getConnection(URL);
            return conn;
        }catch (Exception e){
            System.out.println("DB Connection Failed");
            e.printStackTrace();
        }
        return null;
    }
}
